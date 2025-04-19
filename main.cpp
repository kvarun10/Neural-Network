#include "neural_network.h"
#include <iostream>
using namespace std;

int main() {
    NeuralNetwork nn(2, 4); // XOR: 2 input features, 4 hidden neurons

    vector<vector<float>> X = {{0,0}, {0,1}, {1,0}, {1,1}};
    vector<float> y = {0, 1, 1, 0};

    nn.train(X, y, 10000);

    for (auto &sample : X) {
        float pred = nn.predict(sample);
        cout << "Input: [" << sample[0] << ", " << sample[1] << "] => Prediction: " << pred << endl;
    }

    return 0;
}
