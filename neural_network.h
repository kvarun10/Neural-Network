#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H

#include <vector>
using namespace std;

class NeuralNetwork {
    int input_size, hidden_size;
    float learning_rate;
    vector<vector<float>> W1, W2;
    vector<float> hidden, output;

    float sigmoid(float x);
    float sigmoid_derivative(float x);
    float relu(float x);
    float relu_derivative(float x);
    float forward(const vector<float> &x);

public:
    NeuralNetwork(int input, int hidden, float lr = 0.01);
    void train(const vector<vector<float>> &X, const vector<float> &y, int epochs);
    float predict(const vector<float> &x);
};

#endif
