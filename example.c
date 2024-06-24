#include "fann.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#define WINDOWS 14

char** str_split(char* a_str, const char a_delim)
{
    char** result    = 0;
    size_t count     = 0;
    char* tmp        = a_str;
    char* last_comma = 0;
    char delim[2];
    delim[0] = a_delim;
    delim[1] = 0;

    /* Count how many elements will be extracted. */
    while (*tmp)
    {
        if (a_delim == *tmp)
        {
            count++;
            last_comma = tmp;
        }
        tmp++;
    }

    /* Add space for trailing token. */
    count += last_comma < (a_str + strlen(a_str) - 1);

    /* Add space for terminating null string so caller
       knows where the list of returned strings ends. */
    count++;

    result = malloc(sizeof(char*) * count);

    if (result)
    {
        size_t idx  = 0;
        char* token = strtok(a_str, delim);

        while (token)
        {
            assert(idx < count);
            *(result + idx++) = strdup(token);
            token = strtok(0, delim);
        }
        assert(idx == count - 1);
        *(result + idx) = 0;
    }

    return result;
}

void train() {
    const float connection_rate = 1;
    const float learning_rate = 0.0001;
    const unsigned int num_input = WINDOWS * 3;
    const unsigned int num_output = 2;
    const unsigned int num_layers = 3;
    const unsigned int num_neurons_hidden = 30;
    const float desired_error = 0.0001;
    const unsigned int max_iterations = 1000;
    const unsigned int iterations_between_reports = 100;

    struct fann *ann = fann_create_standard(num_layers,
                                   num_input, num_neurons_hidden, num_output);

    fann_train_on_file(ann, "train.data", max_iterations,
                       iterations_between_reports, desired_error);

    // Open test file
    struct fann_train_data *data = fann_read_train_from_file("test.data");

    float error = fann_test_data(ann, data);

    printf("Error: %f\n", error);

    fann_save(ann, "v2x.net");

    fann_destroy(ann);
}

void test() {
    struct fann *ann = fann_create_from_file("v2x.net");

    FILE *file = fopen("test.data", "r");
    if (file == NULL) {
        printf("Error opening file\n");
        return;
    }

    char line[3000];
    fgets(line, sizeof(line), file); // skip first line
    int line_number = 1;
    while (fgets(line, sizeof(line), file)) {
        double input[42];
        double output[2];
        fann_type fann_output[2];
        fann_type fann_input[42];

        if (line_number % 2 == 1) {
            char** tokens = str_split(line, ' ');

            for (int i = 0; *(tokens + i); i++) {
                input[i] = atof(*(tokens + i));
                free(*(tokens + i));
            }
            
            for (int i = 0; i < 42; i++) {
                //printf("%f ", input[i]);
                fann_input[i] = input[i];
            }
            //printf("\n");
        } else {
            char** tokens = str_split(line, ' ');

            for (int i = 0; *(tokens + i); i++) {
                output[i] = atof(*(tokens + i));
                free(*(tokens + i));
            }

            for (int i = 0; i < 2; i++) {
                //printf("%f ", output[i]);
                fann_output[i] = output[i];
            }
            //printf("\n");

            fann_type* calc_out = fann_run(ann, fann_input);
            printf("%f %f,%f %f\n", calc_out[0], calc_out[1], fann_output[0], fann_output[1]);
        }

        line_number++;
    }

    fclose(file);

    fann_destroy(ann);
}

int main()
{
    //train();
    test();

    return 0;
}