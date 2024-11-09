#include <stdio.h>
#include <stdlib.h>
#include <math.h>


// 2.
// Shuffle the dataset
void shuffle(int* array, size_t n)
{
    if (n > 1)
    {
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


// 3,
// Activation function and its derivative
double Sigmoid(double x) { return 1 / (1 + exp(-x)); }
double dSigmoid(double x) { 
    double sig = sigmoid(x);
    return sig * (1 - sig); }


double Softmax(double input) {
    double exp_val = exp(input);
    double softmax_output = exp_val;
    return softmax_output;
}
double softmax_derivative_single_value(double input) {
    double softmax_output = Softmax(input);
    double derivative = softmax_output * (1.0 - softmax_output);
    return derivative;
}


double init_weight() { return ((double)rand()) / ((double)RAND_MAX); }


#define numInputs 16*16
#define numHiddenNodes 128
#define numOutputs 7
#define numTrainingSets 380

int main(void) {

    const double lr = 0.001;     // learning rate

    double hiddenLayer[numHiddenNodes];
    double output[numOutputs];

    double w1[numInputs][numHiddenNodes];
    double w2[numHiddenNodes][numOutputs];

    double training_inputs[numTrainingSets][numInputs] = { {0.0f,0.0f},
                                                      {1.0f,0.0f},
                                                      {0.0f,1.0f},
                                                      {1.0f,1.0f} };
    double training_outputs[numTrainingSets][numOutputs] = { {0.0f},
                                                            {1.0f},
                                                            {1.0f},
                                                            {0.0f} };

    // init
    for (int i = 0; i < numInputs; i++) { // 256
        for (int j = 0; j < numHiddenNodes; j++) { // 128
            w1[i][j] = init_weight();   // w1[256][128]
        }
    }
    for (int i = 0; i < numHiddenNodes; i++) {
        for (int j = 0; j < numOutputs; j++) {
            w2[i][j] = init_weight();   // w2[128][7]
        }
    }



    int trainingSetOrder[] = { 0,1,2,3 };

    int numberOfEpochs = 10000;
    // Train the neural network for a number of epochs
    for (int epochs = 0; epochs < numberOfEpochs; epochs++) {

        // As per SGD, shuffle the order of the training set
        shuffle(trainingSetOrder, numTrainingSets);

        // Cycle through each of the training set elements
        for (int x = 0; x < numTrainingSets; x++) {

            int i = trainingSetOrder[x];

            // Forward pass

            // Compute hidden layer activation
            for (int j = 0; j < numHiddenNodes; j++) {
                double activation[numHiddenNodes];
                for (int k = 0; k < numInputs; k++) {
                    activation[j] += training_inputs[i][k] * w1[k][j];
                }
                hiddenLayer[j] = Sigmoid(activation[j]);
            }

            // Compute output layer activation
            for (int j = 0; j < numOutputs; j++) {
                double activation[numOutputs];
                for (int k = 0; k < numHiddenNodes; k++) {
                    activation[j] = hiddenLayer[k] * w2[k][j];
                }
                output[j] = Softmax(activation[j]);
            }

            // Print the results from forward pass
            printf("Input:%g  Output:%g    Expected Output: %g\n",
                training_inputs[i][0], training_inputs[i][1],
                output[0], training_outputs[i][0]);



            // Backpropagation

            // Compute change in output weights
            double deltaOutput[numOutputs];
            for (int j = 0; j < numOutputs; j++) {
                double errorOutput = (training_outputs[i][j] - output[j]);
                deltaOutput[j] = errorOutput * dSigmoid(output[j]);
            }

            // Compute change in hidden weights
            double deltaHidden[numHiddenNodes];
            for (int j = 0; j < numHiddenNodes; j++) {
                double errorHidden = 0.0f;
                for (int k = 0; k < numOutputs; k++) {
                    errorHidden += deltaOutput[k] * w2[j][k];
                }
                deltaHidden[j] = errorHidden * dSoftmax(hiddenLayer[j]);
            }

            // Apply change in output weights
            for (int j = 0; j < numOutputs; j++) {
                for (int k = 0; k < numHiddenNodes; k++) {
                    w2[k][j] = w2[k][j] + (hiddenLayer[k] * deltaOutput[j] * lr);
                }
            }

            // Apply change in hidden weights
            for (int j = 0; j < numHiddenNodes; j++) {
                for (int k = 0; k < numInputs; k++) {
                    w1[k][j] = w1[k][j] + (training_inputs[i][k] * deltaHidden[j] * lr);
                }
            }
        }
    }

    // Print final weights after training
    fputs("Final Hidden Weights\n[ ", stdout);
    for (int j = 0; j < numHiddenNodes; j++) {
        fputs("[ ", stdout);
        for (int k = 0; k < numInputs; k++) {
            printf("%f ", w1[k][j]);
        }
        fputs("] ", stdout);
    }


    fputs("]\nFinal Output Weights", stdout);
    for (int j = 0; j < numOutputs; j++) {
        fputs("[ ", stdout);
        for (int k = 0; k < numHiddenNodes; k++) {
            printf("%f ", w2[k][j]);
        }
        fputs("]\n", stdout);
    }


    fputs("]\n", stdout);

    return 0;
}