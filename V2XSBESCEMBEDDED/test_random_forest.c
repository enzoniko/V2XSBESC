#include <stdio.h>
#include <stdlib.h>
#include "exported/RF/rf1.h"  // Replace with actual header name for rf1
#include "exported/RF/rf2.h"  // Replace with actual header name for rf2

int main(int argc, const char *argv[]) {
    if (argc != 1) {
        fprintf(stderr, "Usage: %s\n", argv[0]);
        return -1;
    }

    // Open X_test.txt and Y_test.txt files
    FILE *X_test_file = fopen("exported/RF/X_test.txt", "r");
    FILE *Y_test_file = fopen("exported/RF/Y_test.txt", "r");

    if (X_test_file == NULL || Y_test_file == NULL) {
        perror("Error opening input files");
        return -1;
    }

    // Read number of samples from X_test.txt (assuming both X_test.txt and Y_test.txt have same number of lines)
    int num_samples = 0;
    char line[256];  // Adjust buffer size as needed

    while (fgets(line, sizeof(line), X_test_file)) {
        num_samples++;
    }

    rewind(X_test_file);  // Reset file pointer to start of file

    // Arrays to store data from files
    float X_test[num_samples][3];  // Adjust dimensions based on your X_test.txt format
    float Y_test[num_samples][2];  // Adjust dimensions based on your Y_test.txt format

    // Read X_test.txt and Y_test.txt data into arrays
    for (int i = 0; i < num_samples; i++) {
        // Read X_test.txt data
        for (int j = 0; j < 3; j++) {
            if (fscanf(X_test_file, "%f", &X_test[i][j]) != 1) {
                fprintf(stderr, "Error reading X_test.txt at line %d\n", i + 1);
                return -1;
            }
        }
        
        // Read Y_test.txt data
        for (int j = 0; j < 2; j++) {
            if (fscanf(Y_test_file, "%f", &Y_test[i][j]) != 1) {
                fprintf(stderr, "Error reading Y_test.txt at line %d\n", i + 1);
                return -1;
            }
        }
        
        // Check for end of line in X_test.txt
        char newline;
        if (fscanf(X_test_file, "%c", &newline) != EOF && newline != '\n') {
            fprintf(stderr, "Error: Extra data found in X_test.txt at line %d\n", i + 1);
            return -1;
        }
        
        // Check for end of line in Y_test.txt
        if (fscanf(Y_test_file, "%c", &newline) != EOF && newline != '\n') {
            fprintf(stderr, "Error: Extra data found in Y_test.txt at line %d\n", i + 1);
            return -1;
        }
    }


    // Close files after reading
    fclose(X_test_file);
    fclose(Y_test_file);

    // Variables to store predictions and calculate MAE
    float Y_pred[num_samples][2];
    float mae1 = 0.0, mae2 = 0.0;

    // Predict using rf1 and rf2 models and calculate MAE
    for (int i = 0; i < num_samples; i++) {
        // Predict using rf1 model
        const float features1[] = { X_test[i][0], X_test[i][1], X_test[i][2], X_test[i][3], X_test[i][4], X_test[i][5] };
        float prediction1 = eml_trees_regress1(&rf1, features1, 3);
        Y_pred[i][0] = prediction1;

        // Predict using rf2 model
        const float features2[] = { X_test[i][0], X_test[i][1], X_test[i][2], X_test[i][3], X_test[i][4], X_test[i][5] };
        float prediction2 = eml_trees_regress1(&rf2, features2, 3);
        Y_pred[i][1] = prediction2;

        // Calculate MAE for each feature
        mae1 += fabs(Y_test[i][0] - prediction1);
        mae2 += fabs(Y_test[i][1] - prediction2);
    }

    // Calculate average MAE
    mae1 /= num_samples;
    mae2 /= num_samples;

    // Print MAE for evaluation
    printf("MAE for feature 1: %.4f\n", mae1);
    printf("MAE for feature 2: %.4f\n", mae2);

    return 0;
}