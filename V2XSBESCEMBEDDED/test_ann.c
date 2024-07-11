#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define INPUT_SIZE 6
#define HIDDEN_UNITS 10
#define OUTPUT_SIZE 2

// Function to apply ReLU activation
void relu(double *input, int size) {
    for (int i = 0; i < size; ++i) {
        if (input[i] < 0) {
            input[i] = 0;
        }
    }
}

// Function to perform matrix-vector multiplication
void matvec_mult(double *result, double *matrix, double *vector, int rows, int cols) {
    for (int i = 0; i < rows; ++i) {
        result[i] = 0.0;
        for (int j = 0; j < cols; ++j) {
            result[i] += matrix[i * cols + j] * vector[j];
        }
    }
}

// Function to load weights from text files
void load_weights(const char *filename, double *weights, int size) {
    FILE *file = fopen(filename, "r");
    if (file == NULL) {
        perror("Error opening file");
        exit(EXIT_FAILURE);
    }

    for (int i = 0; i < size; ++i) {
        if (fscanf(file, "%le", &weights[i]) != 1) {
            perror("Error reading file");
            fclose(file);
            exit(EXIT_FAILURE);
        }
    }
    fclose(file);
}

// Function to load test data from text files
void load_data(const char *filename, double *data, int size) {
    FILE *file = fopen(filename, "r");
    if (file == NULL) {
        perror("Error opening file");
        exit(EXIT_FAILURE);
    }

    for (int i = 0; i < size; ++i) {
        if (fscanf(file, "%le", &data[i]) != 1) {
            perror("Error reading file");
            fclose(file);
            exit(EXIT_FAILURE);
        }
    }
    fclose(file);
}

// ANN forward pass
void ann_forward(double *input, double *output, double *hidden_weights, double *output_weights, double *hidden_biases, double *output_biases) {
    double hidden_layer[HIDDEN_UNITS];
    double output_layer[OUTPUT_SIZE];

    // Compute hidden layer activations
    matvec_mult(hidden_layer, hidden_weights, input, HIDDEN_UNITS, INPUT_SIZE);
    for (int i = 0; i < HIDDEN_UNITS; ++i) {
        hidden_layer[i] += hidden_biases[i];
        // Apply ReLU activation function
        if (hidden_layer[i] < 0) {
            hidden_layer[i] = 0;
        }
    }

    // Compute output layer activations
    matvec_mult(output_layer, output_weights, hidden_layer, OUTPUT_SIZE, HIDDEN_UNITS);
    for (int i = 0; i < OUTPUT_SIZE; ++i) {
        output[i] = output_layer[i] + output_biases[i];
    }

    // Debug print statements
    /* printf("Hidden Layer Activations:\n");
    for (int i = 0; i < HIDDEN_UNITS; ++i) {
        printf("%.17f ", hidden_layer[i]);
    }
    printf("\n");

    printf("Output Layer Activations:\n");
    for (int i = 0; i < OUTPUT_SIZE; ++i) {
        printf("%.17f ", output_layer[i]);
    }
    printf("\n");

    printf("Final Output Predictions:\n");
    for (int i = 0; i < OUTPUT_SIZE; ++i) {
        printf("%.17f ", output[i]);
    }
    printf("\n"); */
}

// Function to calculate Mean Absolute Error (MAE)
double calculate_mae(double *predictions, double *actuals, int num_samples) {
    double mae = 0.0;
    for (int i = 0; i < num_samples * OUTPUT_SIZE; ++i) {
        //int idx = 2 * i;
        mae += fabs(predictions[i] - actuals[i]);
    }
    return mae / (num_samples * OUTPUT_SIZE);
}

int main() {
    // Load test data
    int num_samples = 5359;  // Update this to the actual number of samples
    double X_test[num_samples * INPUT_SIZE];
    double Y_test[num_samples * OUTPUT_SIZE];
    load_data("exported/ANN/X_test.txt", X_test, num_samples * INPUT_SIZE);
    load_data("exported/ANN/Y_test.txt", Y_test, num_samples * OUTPUT_SIZE);

    // Allocate memory for weights and biases
    double hidden_weights[HIDDEN_UNITS * INPUT_SIZE];
    double output_weights[OUTPUT_SIZE * HIDDEN_UNITS];
    double hidden_biases[HIDDEN_UNITS];
    double output_biases[OUTPUT_SIZE];

    // Load weights and biases
    load_weights("exported/ANN/dense_weight_0.txt", hidden_weights, HIDDEN_UNITS * INPUT_SIZE);
    load_weights("exported/ANN/dense_weight_1.txt", hidden_biases, HIDDEN_UNITS);
    load_weights("exported/ANN/dense_1_weight_0.txt", output_weights, OUTPUT_SIZE * HIDDEN_UNITS);
    load_weights("exported/ANN/dense_1_weight_1.txt", output_biases, OUTPUT_SIZE);

    // Allocate memory for predictions
    double predictions[num_samples * OUTPUT_SIZE];

    // Perform inference on all test samples
    for (int i = 0; i < num_samples; ++i) {
        ann_forward(&X_test[i * INPUT_SIZE], &predictions[i * OUTPUT_SIZE], hidden_weights, output_weights, hidden_biases, output_biases);
    }

    // Calculate MAE
    double mae = calculate_mae(predictions, Y_test, num_samples);
    printf("Mean Absolute Error: %.17e\n", mae);

    /* for (int i = 0; i < 1; ++i) {
        printf("Actual: %.17e Predicted: %.17e\n", Y_test[i], predictions[i]);
    } */

    return 0;
}