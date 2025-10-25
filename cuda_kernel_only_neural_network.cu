#include <stdio.h>
#include <stdlib.h>
#include <curand_kernel.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

// iris_io.h (drop into your .cu/.cpp)
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <random>
#include <stdexcept>
#include <typeinfo>


// Reads CSV with columns: sepal_length,sepal_width,petal_length,petal_width,species
// Produces row-major X (N*4) and one-hot Y (N*3)
static inline void load_iris_csv(const std::string& path,
                                 std::vector<float>& X,   // (N*4)
                                 std::vector<float>& Y) { // (N*3)
    std::ifstream f(path);
    if (!f) throw std::runtime_error("Could not open: " + path);

    std::string line;
    // Try to detect header (peek first line for non-digit)
    std::getline(f, line);
    auto looks_header = line.find_first_of("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ") != std::string::npos;
    if (!looks_header) {
        // rewind to start of file if first line is data
        f.clear();
        f.seekg(0);
    }

    std::vector<std::array<float,4>> feats;
    std::vector<int> labels;

    while (std::getline(f, line)) {
        if (line.empty()) continue;
        std::stringstream ss(line);
        std::string tok;
        float vals[4]; int k=0;
        for (; k<4 && std::getline(ss, tok, ','); ++k) vals[k] = std::stof(tok);
        if (k < 4) continue; // malformed
        std::string species;
        if (!std::getline(ss, species, ',')) {
            // some files may not have trailing comma; read rest of line
            species = "";
        }
        // if there are extra commas (e.g., whitespace), trim
        species.erase(0, species.find_first_not_of(" \t\r\n"));
        species.erase(species.find_last_not_of(" \t\r\n") + 1);

        feats.push_back({vals[0], vals[1], vals[2], vals[3]});
//labels.push_back(species_to_idx(species));
	printf("%s\n", typeid(species).name());
	labels.push_back(std::stoi(species));

    }

    const int N = static_cast<int>(feats.size());
    const int D = 4, C = 3;
    X.resize(N * D);
    Y.assign(N * C, 0.0f);

    for (int i = 0; i < N; ++i) {
        // row-major: X[i*D + j]
        for (int j = 0; j < D; ++j) X[i*D + j] = feats[i][j];
        Y[i*C + labels[i]] = 1.0f; // one-hot
    }
}

// Optional: standardize features per column (mean 0, std 1)
static inline void standardize_inplace(std::vector<float>& X, int N, int D, float eps=1e-6f) {
    for (int j = 0; j < D; ++j) {
        double sum=0.0, sq=0.0;
        for (int i = 0; i < N; ++i) { float v = X[i*D + j]; sum += v; sq += v*v; }
        double mu = sum / N;
        double var = std::max(0.0, sq / N - mu*mu);
        double sd = std::sqrt(var + eps);
        for (int i = 0; i < N; ++i) X[i*D + j] = static_cast<float>((X[i*D + j] - mu) / sd);
    }
}

__global__ void mat_mul(float *a, float *b, float *c, int m, int n, int k) {
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	float store = 0.0f;
	if (row < m && col < k) {
		for (int i = 0; i < n; i++) {
			store += a[row * n + i] * b[i * k + col];
		}
		c[row * k + col] = store;
	}	
}	

__global__ void relu_inplace(float *x, int n){
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if (i < n) x[i] = x[i] > 0.f ? x[i] : 0.f;
}


__global__ void softmax(int w, int h, float* input, float* output)
{
  int col = blockIdx.x*blockDim.x + threadIdx.x;
  int row = blockIdx.y*blockDim.y + threadIdx.y;
  if (row < h && col < w)
  {
    float maxval = input[row*w];
    for (int i = 1; i<w; i++)
    {
      maxval = fmaxf(maxval, input[row*w + i]);
    }
    float divisor = 0.f;
    for (int i = 0; i<w; i++)
    {
      divisor += expf(input[row*w + i] - maxval);
    }
    output[row*w + col] = expf(input[row*w + col]-maxval)/(divisor);
  }
}

__global__ void cross_entropy(int w, int h, float* preds, float* real, float* output)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < h)
    {
        float loss = 0.f;
        for (int i = 0; i < w; i++)
        {
            loss -= real[idx * w + i] * logf(fmaxf(1e-6, preds[idx * w + i]));
        }
        output[idx] = loss;
    }
}

__global__ void init_weights(float *a, int m, int n)  {
	int col = blockIdx.x*blockDim.x + threadIdx.x;
        int row = blockIdx.y*blockDim.y + threadIdx.y;
	if (row < m && col < n) {
		curandState r;
		curand_init(42, row * n + col, 0, &r);
		a[row * n + col] = curand_normal(&r) * sqrtf(2.f / m);
	}
}	

__global__ void cross_entropy_backwards(float *pred, float *real, float *out, int batch, int cols) {
	int row = blockDim.y * blockIdx.y + threadIdx.y;
	int col = blockDim.x * blockIdx.x + threadIdx.x;

	if (row < batch && col < cols) {
		out[row * cols + col] = pred[row * cols + col] - real[row * cols + col];
	}
}	

__global__ void backwards(float *cur_deriv, float *cur_weight, float *output,
                          int batch, int next_dim, int cur_dim) {
    int row = blockDim.y * blockIdx.y + threadIdx.y; // 0..batch-1
    int col = blockDim.x * blockIdx.x + threadIdx.x; // 0..cur_dim-1

    if (row < batch && col < cur_dim) {
        float acc = 0.0f;
        // dA[row, col] = sum_k dZ[row, k] * W[col, k]
        for (int k = 0; k < next_dim; ++k) {
            float dz = cur_deriv[row * next_dim + k];        // dZ[row, k]
            float wk = cur_weight[col * next_dim + k];       // W[col, k]
            acc += dz * wk;
        }
        output[row * cur_dim + col] = acc; // dA[row, col]
    }
}

__global__ void activation_grad(float *cur_grad, float *a_out, float *output, int next_dim, int batch) {
	int row = blockDim.y * blockIdx.y + threadIdx.y;
	int col = blockDim.x * blockIdx.x + threadIdx.x;
	if (row < batch && col < next_dim) {
		output[row * next_dim + col] = a_out[row * next_dim + col] > 0 ? cur_grad[row * next_dim + col] : 0;
	}
}

// accumulate dW and db
__global__ void dW_db(const float* __restrict__ A,   // (B,in)
                      const float* __restrict__ dZ,  // (B,out)
                      float* __restrict__ dW,        // (in,out)
                      float* __restrict__ db,        // (out)
                      int B,int in,int out){
  int r = blockIdx.y*blockDim.y + threadIdx.y; // in
  int c = blockIdx.x*blockDim.x + threadIdx.x; // out
  if (r < in && c < out){
    float acc = 0.f;
    for (int b=0;b<B;++b) acc += A[b*in + r] * dZ[b*out + c];
    dW[r*out + c] = acc;
  }
  if (r==0 && c<out){
    float accb = 0.f;
    for (int b=0;b<B;++b) accb += dZ[b*out + c];
    db[c] = accb;
  }
}

__global__ void apply_sgd(float* __restrict__ W,const float* __restrict__ dW,
                          float* __restrict__ b,const float* __restrict__ db,
                          float lr,int B,int in,int out){
  int r = blockIdx.y*blockDim.y + threadIdx.y; // in
  int c = blockIdx.x*blockDim.x + threadIdx.x; // out
  if (r < in && c < out) {
    W[r*out + c] -= lr * (dW[r*out + c] / B);
  }
  if (r == 0 && c < out) {
	  b[c] -= lr * (db[c] / B);
  }
}



__global__ void sgd(float* weights, float *biases, float *activations, float *d_l, float lr, int batch_size, int m, int n) {
	int row = blockDim.y * blockIdx.y + threadIdx.y;
	int col = blockDim.x * blockIdx.x + threadIdx.x;
	if (row < m && col < n) {
		float dw = 0.0f;
		float db = 0.0f;
		for (int i = 0; i < batch_size; i++) {
			float act =  activations[i * m + row];
			float dl = d_l[i * n + col];
			dw += act * dl;
			db += dl;
		}
	weights[row * n + col] -= lr * dw / batch_size;
	biases[col] -= lr * db / batch_size;
	}
}

struct Batch {
    const float* X;  // points into row-major X (N*D)
    const float* Y;  // points into row-major Y (N*C)
    int size;        // number of rows in this batch
};

struct Batcher {
    const std::vector<float>& X;
    const std::vector<float>& Y;
    int N, D, C, bs, pos = 0;

    Batcher(const std::vector<float>& X_, const std::vector<float>& Y_,
            int N_, int D_, int C_, int bs_)
        : X(X_), Y(Y_), N(N_), D(D_), C(C_), bs(bs_) {}

    inline bool next(Batch& out) {
        if (pos >= N) return false;
        int n = std::min(bs, N - pos);
        out = Batch{ &X[pos * D], &Y[pos * C], n };
        pos += n;
        return true;
    }

    inline void reset() { pos = 0; }
};



int main() {

	std::vector<float> X, Y;
	load_iris_csv("./iris.csv", X, Y);
	int N = Y.size() / 3;
	int D = 4;
	int C = 3;
	standardize_inplace(X, N, D);
	int Ntrain = (int)(0.8 * N);
	int H = 16;
        float lr = 0.1f;
	float *d_W1, *d_b1, *d_W2, *d_b2;
        cudaMalloc(&d_W1, D*H*sizeof(float));
        cudaMalloc(&d_b1, H*sizeof(float));
        cudaMalloc(&d_W2, H*C*sizeof(float));
        cudaMalloc(&d_b2, C*sizeof(float));
	dim3 b(16,16);
        dim3 g1((H+15)/16, (D+15)/16), g2((C+15)/16, (H+15)/16);
        
	init_weights<<<g1,b>>>(d_W1, D, H);
        init_weights<<<g2,b>>>(d_W2, H, C);

	cudaMemset(d_b1, 0, H*sizeof(float));
        cudaMemset(d_b2, 0, C*sizeof(float));

	float *d_X,*d_Y,*d_Z1,*d_A1,*d_Z2,*d_P,*d_loss;
	cudaMalloc(&d_X, Ntrain*D*sizeof(float));
    	cudaMalloc(&d_Y, Ntrain*C*sizeof(float));
        cudaMalloc(&d_Z1, Ntrain*H*sizeof(float));
	cudaMalloc(&d_A1, Ntrain*H*sizeof(float));
        cudaMalloc(&d_Z2, Ntrain*C*sizeof(float));
	cudaMalloc(&d_P,  Ntrain*C*sizeof(float));
        cudaMalloc(&d_loss, Ntrain*sizeof(float));

	cudaMemcpy(d_X, X.data(), Ntrain*D*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_Y, Y.data(), Ntrain*C*sizeof(float), cudaMemcpyHostToDevice);

	float *d_dZ2,*d_dA1,*d_dW1,*d_db1,*d_dW2,*d_db2;
	cudaMalloc(&d_dZ2, Ntrain*C*sizeof(float));
        cudaMalloc(&d_dA1, Ntrain*H*sizeof(float));
	cudaMalloc(&d_dW1, D*H*sizeof(float));
        cudaMalloc(&d_db1, H*sizeof(float));
	cudaMalloc(&d_dW2, H*C*sizeof(float));
	cudaMalloc(&d_db2, C*sizeof(float));

	for (int i = 0; i < 10; i++) {
		mat_mul<<<g1,b>>>(d_X,d_W1,d_Z1,Ntrain,D,H);
		relu_inplace<<<1, Ntrain * H>>>(d_Z1,Ntrain*H); 
		cudaMemcpy(d_A1,d_Z1,Ntrain*H*sizeof(float),cudaMemcpyDeviceToDevice);
		mat_mul<<<g2,b>>>(d_A1,d_W2,d_Z2,Ntrain,H,C);
		softmax<<<g2,b>>>(C,Ntrain,d_Z2,d_P);
		cross_entropy<<<1, Ntrain>>>(C,Ntrain,d_P,d_Y,d_loss);

		cross_entropy_backwards<<<g2,b>>>(d_P,d_Y,d_dZ2,Ntrain,C);
		dW_db<<<g2,b>>>(d_A1,d_dZ2,d_dW2,d_db2,Ntrain,H,C);
		backwards<<<g1,b>>>(d_dZ2,d_W2,d_dA1,Ntrain,C,H);
		activation_grad<<<g1,b>>>(d_dA1,d_A1,d_dA1,H,Ntrain);
		dW_db<<<g1,b>>>(d_X,d_dA1,d_dW1,d_db1,Ntrain,D,H);
		apply_sgd<<<g1,b>>>(d_W1,d_dW1,d_b1,d_db1,lr,Ntrain,D,H);
		apply_sgd<<<g2,b>>>(d_W2,d_dW2,d_b2,d_db2,lr,Ntrain,H,C);
		cudaDeviceSynchronize();
		printf("Epoch %d done.\n", i+1);
	}
	cudaFree(d_X); cudaFree(d_Y);	
}


