#define STB_IMAGE_IMPLEMENTATION
#include <stdio.h>
#include "stb_image.h"
#include <stdlib.h>
#include <math.h>
#include <time.h>


#define MAX_ITERATION 100
#define SIZE 28
#define DATA_COUNT_EACH 100 //200 data total, 100 of each class - 160 in training set since %80
#define SEED_1 12
#define SEED_2 24
#define SEED_3 32232134

#define BETA_1 0.9
#define BETA_2 0.999
#define EPSILON 1e-8


typedef struct{
    double* input;
    int output;
}Data;

void clipGradient(double* gradient, int size, double clip_threshold) {
    for (int i = 0; i < size; i++) {
        if (gradient[i] > clip_threshold) {
            gradient[i] = clip_threshold;
        } else if (gradient[i] < -clip_threshold) {
            gradient[i] = -clip_threshold;
        }
    }
}


void copyArray(double* source,double* target,int size)
{
    for(int i = 0; i< size ;i++){
        target[i]= source[i];
    }
}

void saveResult(double* source, double* target,int iteration,double time,double successRate)
{
    target[0] = iteration;
    target[1] = time;
    target[2] = successRate;
    for(int i = 3; i<SIZE*SIZE+4 ;i++){
        target[i]= source[i-3];
    }
}

double distanceOfArrays(double* first, double* second, int size)
{
    double result = 0;
    for(int i = 0 ; i < size;i++){
        result += pow(first[i]-second[i],2);
    }
    return sqrt(result);
}

void imageToSquareArray(const char *path, double *array)
{
    int width, height, channels;
    unsigned char *image_data = stbi_load(path, &width, &height, &channels, 1);
    printf("%i",width);
    if(width != SIZE || height != SIZE)
    {
        printf("Invalid image size!");
        return;
    }
    for(int i = 0; i < SIZE*SIZE ;i++)
    {
            //printf("array[%d] = %f\n", i, (double)image_data[i]/255);
            array[i] = ((double)image_data[i]/255);
            printf("array[%d] = %f\n",i,array[i]);
    }
    stbi_image_free(image_data);
}

void prepareData(const char* pathToMainFolder,Data* trainingSet, Data* testingSet)
{

    char* firstClassName = "A";
    char* secondClassName = "B";
    int randomInts[DATA_COUNT_EACH];
    int testCount = 0;
    int trainCount = 0;
    generateRandomNums(randomInts,SEED_1,DATA_COUNT_EACH);
    for(int i = 0; i < DATA_COUNT_EACH;i++){
        char path[256];
        snprintf(path, sizeof(path), "Images/%s/%d.png", firstClassName, randomInts[i]);
        /*double** array = (double**)malloc(SIZE * sizeof(double*));
        for (int j = 0; j < SIZE; j++)
            array[j] = (double*)malloc(SIZE * sizeof(double));
            */
        //double* vec = (double*)malloc((SIZE*SIZE+1)*sizeof(double));
        //imageToSquareArray(path,vec);
        //vec[SIZE*SIZE] = 1;
        if(testCount<20){
            testingSet[testCount].input = (double*)malloc((SIZE*SIZE+1)*sizeof(double));
            imageToSquareArray(path,testingSet[testCount].input);
            testingSet[testCount].input[SIZE*SIZE] = 1;
            testingSet[testCount].output = 1;

            testCount++;
        }
        else{
            trainingSet[trainCount].input = (double*)malloc((SIZE*SIZE+1)*sizeof(double));
            imageToSquareArray(path,trainingSet[trainCount].input);
            trainingSet[trainCount].input[SIZE*SIZE] = 1;
            trainingSet[trainCount].output = 1;
            trainCount++;
        }
    }
    generateRandomNums(randomInts,SEED_2,DATA_COUNT_EACH);
    for(int i = 0; i < DATA_COUNT_EACH;i++){
        char path[256];
        snprintf(path, sizeof(path), "Images/%s/%d.png", secondClassName, randomInts[i]);
        /*double** array = (double**)malloc(SIZE * sizeof(double*));
        for (int j = 0; j < SIZE; j++)
            array[j] = (double*)malloc(SIZE * sizeof(double));*/
        double* vec = (double*)malloc((SIZE*SIZE+1)*sizeof(double));
        imageToSquareArray(path,vec);
        vec[SIZE*SIZE] = 1;
        if(testCount<40){
            testingSet[testCount].input = (double*)malloc((SIZE*SIZE+1)*sizeof(double));
            imageToSquareArray(path,testingSet[testCount].input);
            testingSet[testCount].input[SIZE*SIZE] = 1;
            testingSet[testCount].output = -1;
            testCount++;
        }
        else{
            trainingSet[trainCount].input = (double*)malloc((SIZE*SIZE+1)*sizeof(double));
            imageToSquareArray(path,trainingSet[trainCount].input);
            trainingSet[trainCount].input[SIZE*SIZE] = 1;
            trainingSet[trainCount].output = -1;
            trainCount++;
        }
    }
}

void generateRandomNums2(int* output,int seed){ //eski kod
    srand(seed);
    for (int i = 0;i<DATA_COUNT_EACH;i++){
     if(i<DATA_COUNT_EACH/5.0){
          output[i] = i;
      }
      else{
        int random = rand()%(i+1);
        if(random < DATA_COUNT_EACH/5.0){
            output[random] = i;
        }
      }
    }
}

void generateRandomNums(int* arr,int seed,int size){
    srand(seed);
    for(int i = 0;i<size;i++){
        arr[i] = i;
    }
    for (int i = size-1; i > 0; i--) {
        int j = rand() % (i + 1);
        int temp = arr[i];
        arr[i] = arr[j];
        arr[j] = temp;
    }
}


int containsThisNum(int num, int* array) //eski kod
{
    for(int i = 0; i <  DATA_COUNT_EACH/5.0;i++){
        if(num == array[i]){
            return 1;
        }
    }
    return 0;

}

void reshapeToFlatPlusOne(double** input,double* output){
    for(int i = 0 ;i < SIZE;i++){
        for(int j = 0;j < SIZE;j++){
            output[SIZE*i+j] = input[i][j];
        }
    }
    output[SIZE*SIZE] = 1;
}



double func(double* w, double* x){
    double result = 0;
    for(int j = 0;j<SIZE*SIZE+1;j++){
        result += x[j] * w[j];
    }
    return tanh(result);
}

double testResults(double* parameters,Data* testSet)
{
    double guess;
    int correctGuesses = 0;
    for(int j = 0;j < DATA_COUNT_EACH*(2.0/5.0);j++)
    {
        guess = func(parameters,testSet[j].input);
        if(guess*testSet[j].output>0){
            correctGuesses++;
        }
    }
    return (double)(correctGuesses)/(double)(DATA_COUNT_EACH*(2.0/5.0));
}

double derOfFunc(double* w,double* x,int index){
    return x[index]*(1-pow(func(w,x),2));
}

double lossFuncTotal(double* w,Data* trainingSet){
    double result = 0;
    for(int i = 0;i<DATA_COUNT_EACH*(8.0/5.0);i++){
         result += pow((func(w,trainingSet[i].input)-trainingSet[i].output),2);
    }
    return result/(DATA_COUNT_EACH*(8.0/5.0));
}

void derOfLossFuncTotal(double* w, Data* trainingSet,double* derivatives){
    for(int j = 0;j<SIZE*SIZE+1;j++){
        double result = 0;
        for(int i = 0;i<DATA_COUNT_EACH*(8.0/5.0);i++){
            result += 2*(func(w,trainingSet[i].input)-trainingSet[i].output)*derOfFunc(w,trainingSet[i].input,j);
        }
        derivatives[j] = result/(DATA_COUNT_EACH*(8.0/5.0));
    }
}

double lossOfData(double* w,Data* data){
    return pow(data->output - func(w,data->input),2);//yi-ypredict
}

void derOfLossOfData(double* w,Data* data,double* derivatives){
    for(int j = 0;j<SIZE*SIZE+1;j++){
         double result = -2*(data->output - func(w,data->input)) * derOfFunc(w,data->input, j);
        derivatives[j] = result;
    }
}


void gradientDescent(double* startingParameters,Data* trainingSet,double stepSize,double precision, Data* testSet,int saveNum)
{
    char buffer[100];
    sprintf(buffer,"C:/gd_start%i.csv",saveNum);
    FILE *file = fopen(buffer, "a");
    if (file == NULL) {
        perror("Error opening file");
        return;
    }
    double new_w[SIZE*SIZE+1];
    double old_w[SIZE*SIZE+1];
    double gradient[SIZE*SIZE+1];
    copyArray(startingParameters,old_w,SIZE*SIZE+1);
    int iteration = 0;
    double totalTime = 0;
    printf("starting success");
    double success = testResults(old_w,testSet);
    printf("got through success");
    fprintf(file,"%i,%f,%f,%f",iteration,0.0,success,lossFuncTotal(old_w,trainingSet));
    for(int i = 0; i < SIZE*SIZE+1;i++)
    {
        fprintf(file,",%f",old_w[i]);
    }
    fprintf(file,"\n");

    while(iteration < MAX_ITERATION)
    {
        iteration++;
        clock_t start = clock();
        derOfLossFuncTotal(old_w,trainingSet,gradient);
        for(int i = 0; i < SIZE*SIZE+1;i++){
            new_w[i] = old_w[i] - stepSize * gradient[i];
        }
        //write to file here
        copyArray(new_w,old_w,SIZE*SIZE+1);
        totalTime += (double)(clock()-start);
        double success = testResults(old_w,testSet);
        double loss = lossFuncTotal(old_w,trainingSet);
        fprintf(file,"%i,%f,%f,%f",iteration,totalTime,success,loss);
        for(int i = 0; i < SIZE*SIZE+1;i++)
        {
            fprintf(file,",%f",old_w[i]);
        }
        fprintf(file,"\n");
        printf("%-i iteration:%f-%f\n",iteration,success,loss);
        /*if(distanceOfArrays(new_w,old_w,SIZE*SIZE+1) < precision){
            return;
        }
        */

    }
    fclose(file);
}

void stochasticGradientDescent(double* startingParameters,Data* trainingSet,double stepSize,double precision, Data* testSet,int saveNum)
{
    char buffer[100];
    sprintf(buffer,"C:/sgd_start%i.csv",saveNum);
    FILE *file = fopen(buffer, "a");
    if (file == NULL) {
        perror("Error opening file");
        return;
    }

    double new_w[SIZE*SIZE+1];
    double old_w[SIZE*SIZE+1];
    double gradient[SIZE*SIZE+1];

    int randomNums[(int)(DATA_COUNT_EACH*(8.0/5.0))];
    int randomCount = 0;
    generateRandomNums(randomNums,time(NULL),(int)(DATA_COUNT_EACH*(8.0/5.0)));

    copyArray(startingParameters,old_w,SIZE*SIZE+1);
    int iteration = 0;
    double totalTime = 0;
    printf("starting success");
    double success = testResults(old_w,testSet);
    printf("got through success");
    fprintf(file,"%i,%f,%f,%f",iteration,0.0,success,lossFuncTotal(old_w,trainingSet));
    for(int i = 0; i < SIZE*SIZE+1;i++)
    {
        fprintf(file,",%f",old_w[i]);
    }
    fprintf(file,"\n");

    while(iteration < MAX_ITERATION)
    {
        iteration++;
        clock_t start = clock();
        if(randomCount >= (int)(DATA_COUNT_EACH*(8.0/5.0))){
            generateRandomNums(randomNums,time(NULL),(int)(DATA_COUNT_EACH*(8.0/5.0)));
        randomCount = 0;
        }
        derOfLossOfData(old_w,&trainingSet[randomNums[randomCount]],gradient);
        clipGradient(gradient,(int)(DATA_COUNT_EACH*(8.0/5.0)),1.0);
        randomCount++;

        for(int i = 0; i < SIZE*SIZE+1;i++){
            new_w[i] = old_w[i] - stepSize * (1-(iteration/MAX_ITERATION)) * gradient[i];
        }
        copyArray(new_w,old_w,SIZE*SIZE+1);
        totalTime += (double)(clock()-start);
        double success = testResults(old_w,testSet);
        double loss = lossFuncTotal(old_w,trainingSet);
        fprintf(file,"%i,%f,%f,%f",iteration,totalTime,success,loss);
        for(int i = 0; i < SIZE*SIZE+1;i++)
        {
            fprintf(file,",%f",old_w[i]);
        }
        fprintf(file,"\n");
        printf("%-i iteration:%f-%f\n",iteration,success,loss);
        /*if(distanceOfArrays(new_w,old_w,SIZE*SIZE+1) < precision){
            return;
        }
        */

    }
    fclose(file);
}

void adam(double* startingParameters,Data* trainingSet,double stepSize,double precision, Data* testSet,int saveNum){

    char buffer[100];
    sprintf(buffer,"C:/adam_start%i.csv",saveNum);
    FILE *file = fopen(buffer, "a");
    if (file == NULL) {
        perror("Error opening file");
        return;
    }

    double new_w[SIZE*SIZE + 1];
    double old_w[SIZE*SIZE + 1];
    double gradient[SIZE*SIZE + 1];
    double m[SIZE*SIZE + 1] = {0};
    double v[SIZE*SIZE + 1] = {0};
    double mHat[SIZE*SIZE + 1] = {0};
    double vHat[SIZE*SIZE + 1] = {0};

    int randomNums[(int)(DATA_COUNT_EACH*(8.0/5.0))];
    int randomCount = 0;
    generateRandomNums(randomNums,time(NULL),(int)(DATA_COUNT_EACH*(8.0/5.0)));


    copyArray(startingParameters, old_w, SIZE*SIZE + 1);

    int iteration = 0;
    double totalTime = 0;

    double success = testResults(old_w, testSet);

    fprintf(file, "%i,%f,%f,%f", iteration, 0.0, success, lossFuncTotal(old_w, trainingSet));
    for (int i = 0; i < SIZE*SIZE + 1; i++) {
        fprintf(file, ",%f", old_w[i]);
    }
    fprintf(file, "\n");

    while (iteration < MAX_ITERATION)
    {
        iteration++;
        clock_t start = clock();
        if(randomCount >= (int)(DATA_COUNT_EACH*(8.0/5.0))){
            generateRandomNums(randomNums,time(NULL),(int)(DATA_COUNT_EACH*(8.0/5.0)));
            randomCount = 0;
        }
        derOfLossOfData(old_w,&trainingSet[randomNums[randomCount]],gradient);
        randomCount++;

        for (int i = 0; i < SIZE*SIZE + 1; i++) {
            m[i] = BETA_1 * m[i] + (1 - BETA_1) * gradient[i];
            v[i] = BETA_2 * v[i] + (1 - BETA_2) * gradient[i] * gradient[i];
        }

        for (int i = 0; i < SIZE*SIZE + 1; i++) {
            mHat[i] = m[i] / (1 - pow(BETA_1, iteration));
            vHat[i] = v[i] / (1 - pow(BETA_2, iteration));
        }

        for (int i = 0; i < SIZE*SIZE + 1; i++) {
            new_w[i] = old_w[i] - stepSize * mHat[i] / (sqrt(vHat[i]) + EPSILON);
        }

        copyArray(new_w, old_w, SIZE*SIZE + 1);

        totalTime += (double)(clock() - start);

        double success = testResults(old_w, testSet);
        double loss = lossFuncTotal(old_w, trainingSet);

        fprintf(file, "%i,%f,%f,%f", iteration, totalTime, success, loss);
        for (int i = 0; i < SIZE*SIZE + 1; i++) {
            fprintf(file, ",%f", old_w[i]);
        }
        fprintf(file, "\n");

        printf("%-i iteration:%f-%f\n", iteration, success, loss);

        /*if(distanceOfArrays(new_w,old_w,SIZE*SIZE+1) < precision){
            return;
        }
        */
    }
    fclose(file);
}




int main()
{
    Data trainingSet[(int)(DATA_COUNT_EACH*(8.0/5.0))];
    Data testSet[(int)(DATA_COUNT_EACH*(2.0/5.0))];
    prepareData("C:/Users/MONSTER1/OneDrive/Documents/CodeBlock Projects/",trainingSet,testSet);

    double startingParameters1[SIZE*SIZE+1];
    double startingParameters2[SIZE*SIZE+1];
    double startingParameters3[SIZE*SIZE+1];
    double startingParameters4[SIZE*SIZE+1];
    double startingParameters5[SIZE*SIZE+1];

    for(int i = 0;i < SIZE*SIZE+1;i++){
        startingParameters1[i] = 1;
    }
    for(int i = 0;i < SIZE*SIZE+1;i++){
        startingParameters2[i] = 0.1;
    }
    for(int i = 0;i < SIZE*SIZE+1;i++){
        startingParameters3[i] = 0.01;
    }
    for(int i = 0;i < SIZE*SIZE+1;i++){
        startingParameters4[i] = 0.001;
    }
    for(int i = 0;i < SIZE*SIZE+1;i++){
        startingParameters5[i] = 0.0001;
    }

    //gradientDescent(startingParameters1,trainingSet,0.001,pow(10,-60),testSet,1);
    //gradientDescent(startingParameters2,trainingSet,0.001,pow(10,-60),testSet,2);
    //gradientDescent(startingParameters3,trainingSet,0.001,pow(10,-60),testSet,3);
    //gradientDescent(startingParameters4,trainingSet,0.001,pow(10,-60),testSet,4);
    //gradientDescent(startingParameters5,trainingSet,0.001,pow(10,-60),testSet,5);

    stochasticGradientDescent(startingParameters1,trainingSet,0.01,pow(10,-60),testSet,1);
    stochasticGradientDescent(startingParameters2,trainingSet,0.01,pow(10,-60),testSet,2);
    stochasticGradientDescent(startingParameters3,trainingSet,0.01,pow(10,-60),testSet,3);
    stochasticGradientDescent(startingParameters4,trainingSet,0.01,pow(10,-60),testSet,4);
    stochasticGradientDescent(startingParameters5,trainingSet,0.01,pow(10,-60),testSet,5);

    adam(startingParameters1,trainingSet,0.01,pow(10,-60),testSet,1);
    adam(startingParameters2,trainingSet,0.01,pow(10,-60),testSet,2);
    adam(startingParameters3,trainingSet,0.01,pow(10,-60),testSet,3);
    adam(startingParameters4,trainingSet,0.01,pow(10,-60),testSet,4);
    adam(startingParameters5,trainingSet,0.01,pow(10,-60),testSet,5);


    return 0;
}
