#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>   

#include <string.h>


#define numInputs 256
#define imageWidth 16
#define imageHeight 16

#define numHiddenNodes 128
#define numOutputs 7
#define numTrainingSets 420

#define train_size 420
#define test_size 20

#define batch 128


// 1. Read data
typedef struct {
    int data[16][16];
} Image;

void read_dataset(const char* filename, int image[16][16]) {
    FILE* file = fopen(filename, "r");
    if (!file) {
        printf("파일을 열 수 없습니다: %s\n", filename);
        exit(1);
    }

    // CSV 파일에서 이미지 읽어오기
    for (int i = 0; i < 16; i++) {
        for (int j = 0; j < 16; j++) {
            if (fscanf(file, "%d,", &image[i][j]) != 1) {
                printf("파일에서 데이터를 읽는 중 오류가 발생했습니다.\n");
                exit(1);
            }
        }
    }

    fclose(file);
}

void read_trainlabel(const char* filename, int Y[numTrainingSets]) {
    FILE* file = fopen(filename, "r");
    if (file == NULL)   return;

    char line[1000];
    int index = 0;

    if (fgets(line, sizeof(line), file) != NULL) {
        // Remove newline character at the end of the line
        line[strcspn(line, "\n")] = 0;

        // Tokenize the line using commas
        char* token = strtok(line, ",");
        while (token != NULL && index < numTrainingSets) {
            // Convert string to integer and store in Y array
            Y[index] = atoi(token);
            index++;

            // Move to the next token
            token = strtok(NULL, ",");
        }
    }

    fclose(file);
}
void read_testlabel(const char* filename, int Y[test_size]) {
    FILE* file = fopen(filename, "r");
    if (file == NULL)   return;

    char line[1000];
    int index = 0;

    if (fgets(line, sizeof(line), file) != NULL) {
        // Remove newline character at the end of the line
        line[strcspn(line, "\n")] = 0;

        // Tokenize the line using commas
        char* token = strtok(line, ",");
        while (token != NULL && index < test_size) {
            // Convert string to integer and store in Y array
            Y[index] = atoi(token);
            index++;

            // Move to the next token
            token = strtok(NULL, ",");
        }
    }

    fclose(file);
}


// 2. Shuffle the dataset
void shuffle(int* array, size_t n)
{
    if (n > 1) {
        size_t i;
        for (i = 0; i < n - 1; i++)
        {
            size_t j = i + rand() / (RAND_MAX / (n - i) + 1);
            int t = array[j];
            array[j] = array[i];
            array[i] = t;
        }
    }
}


// 3. define function (sig, softmax, init)
void Sigmoid(double* z, double* output, int size) {
    for (int i = 0; i < size; i++) {
        output[i] = 1.0 / (1.0 + exp(-z[i]));
    }
}
void dSigmoid(double* z, double* output, int size) {
    for (int i = 0; i < size; i++) {
        z[i] = output[i] * (1.0 - output[i]);
    }
}

void Softmax(double *z, double *output, int size) {
    double max_val = z[0];
    for (int i = 1; i < size; i++) {
        if (z[i] > max_val) {
            max_val = z[i];
        }
    }

    double sum_exp = 0.0;
    for (int i = 0; i < size; i++) {
        output[i] = exp(z[i] - max_val);  // 오버플로우 방지를 위해 최대값으로 빼줌
        sum_exp += output[i];
    }

    for (int i = 0; i < size; i++) {
        output[i] /= sum_exp;
    }
}
void dSoftmax(double* z, double* out, double* _targets, int size) {
    for (int i = 0; i < size; i++) {
        z[i] = 2 * (out[i] - _targets[i]) / size;
    }
}

void init1(double layer[numInputs][numHiddenNodes], int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            layer[i][j] = ((double)rand() / RAND_MAX) * 2.0 - 1.0;  // -1부터 1 사이의 난수로 초기화
        }
    }
}
void init2(double layer[numHiddenNodes][numOutputs], int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            layer[i][j] = ((double)rand() / RAND_MAX) * 2.0 - 1.0;  // -1부터 1 사이의 난수로 초기화
        }
    }
}


void flattenMatrix(double* dest, double*** src, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            dest[i * cols + j] = (*src)[i][j];
        }
    }
}
void shuffleArray(int* array, int size) {
    srand((unsigned int)time(NULL));
    for (int i = size - 1; i > 0; i--) {
        int j = rand() % (i + 1);
        // Swap array[i] and array[j]
        int temp = array[i];
        array[i] = array[j];
        array[j] = temp;
    }
}

void dot(double* result, double* matrix1, double* matrix2, int rows1, int cols1, int cols2) {
    for (int i = 0; i < rows1; i++) {
        for (int j = 0; j < cols2; j++) {
            result[i * cols2 + j] = 0;
            for (int k = 0; k < cols1; k++) {
                result[i * cols2 + j] += matrix1[i * cols1 + k] * matrix2[k * cols2 + j];
            }
        }
    }
}
void transposeMatrix(double* result, double* matrix, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            result[j * rows + i] = matrix[i * cols + j];
        }
    }
}

int X_train[420][imageWidth][imageHeight];
int X_test[20][imageWidth][imageHeight];

double w1[numInputs][numHiddenNodes];
double w2[numHiddenNodes][numOutputs];

double d_w1[numInputs][numHiddenNodes];
double d_w2[numHiddenNodes][numOutputs];



int main(void) {

    // 1. read data
    for (int i = 0; i < 420; i++) {
        char filename[256];
        sprintf(filename, "train_dataset_csv/%d.csv", i + 1);
        read_dataset(filename, X_train[i]);
    }

    for (int i = 1; i <= 20; i++) {
        char filename[50];
        snprintf(filename, sizeof(filename), "test_dataset_csv/test%d.csv", i);
        read_dataset(filename, X_test[i - 1]);
    }

    int Y_train[420] = { 0, };
    int Y_test[20] = { 0, };

    read_trainlabel("train_label.csv", Y_train);
    read_testlabel("test_label.csv", Y_test);


   // 3 - (3). Initializing weights
    srand(42);

    // 배열 초기화
    init1(w1, numInputs, numHiddenNodes);
    init2(w2, numHiddenNodes, numOutputs);

    printf("\n%.4f %.4f", w1[255][127], w2[127][6]);

    // 5. Training
    int epochs = 18000;          // num of epochs
    const double lr = 0.001;     // learning rate

    for (int e = 0; e < epochs; e++) {

        double targets[batch][numOutputs] = { 0, };
        double x[batch][numInputs] = { 0, };
        double y[batch] = { 0, };



        int rand[train_size] = { 0, };
        for (int i = 0; i < train_size; i++) {
            rand[i] = i;
        }

        shuffleArray(rand, train_size);

        for (int i = 0; i < batch; i++) {
            int index = rand[i];
            for (int j = 0; j < 16; j++) {
                for (int k = 0; k < 16; k++) {
                    x[i][j * 16 + k] = X_train[index][j][k];
                }
            }
            y[i] = Y_train[index];
        }

        // targets 배열 생성
        for (int i = 0; i < batch; i++) {
            for (int j = 0; j < numOutputs; j++) {
                targets[i][j] = (j == y[i]) ? 1.0 : 0.0;
            }
        }


        // Forward pass
        double z1[batch][numHiddenNodes];
        double a1[batch][numHiddenNodes];
        double z2[batch][numOutputs];
        double output[batch][numOutputs];

        dot(&z1[0][0], &x[0][0], &w1[0][0], batch, numInputs, numHiddenNodes);
        Sigmoid(&z1[0][0], &a1[0][0], batch * numHiddenNodes);
        dot(&z2[0][0], &a1[0][0], &w2[0][0], batch, numHiddenNodes, numOutputs);
        Softmax(&z2[0][0], &output[0][0], batch * numOutputs);


        // Backpropagation
        double error[batch][numOutputs] = { 0, };

        for (int i = 0; i < batch; i++) {
            dSoftmax(&error[i][0], &output[i][0], &targets[i][0], numOutputs);
        }

        transposeMatrix(&a1[0][0], &a1[0][0], batch, numHiddenNodes);
        dot(&d_w2[0][0], &a1[0][0], &error[0][0], numHiddenNodes, batch, numOutputs);

        dot(&error[0][0], &w2[0][0], &error[0][0], batch, numOutputs, numHiddenNodes);
        for (int i = 0; i < batch; i++) {
            dSigmoid(&a1[i][0], &a1[i][0], numHiddenNodes);
        }
        for (int i = 0; i < batch; i++) {
            for (int j = 0; j < numHiddenNodes; j++) {
                error[i][j] *= a1[i][j];
            }
        }
        transposeMatrix(&x[0][0], &x[0][0], batch, numInputs);
        dot(&d_w1[0][0], &x[0][0], &error[0][0], numInputs, batch, numHiddenNodes);

        return 0;
    }
}