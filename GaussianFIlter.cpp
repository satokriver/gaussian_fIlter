#include <iostream>
#include <opencv2/opencv.hpp>

void MyGaussian(const cv::Mat& img, cv::Mat& dst)
{
    std::vector<uchar> img_array;
    img_array.assign(img.data, img.data + img.total() * img.channels());
    //for (int c = 0; c < 10; c++)
    //{
    //    std::cout << (int)img_array[10 * img.cols + c] << std::endl;
    //}

    // https://github.com/ragnraok/android-image-filter/tree/master/library/jni
    int ksize = 5;
    double sigma = 1.0;
    double PI = 3.14159265358979323846264338327;
    double* kernel = new double[ksize * ksize];
    double scale = -0.5 / (sigma * sigma);
    double cons = -scale / PI;
    double sum = 0;

    for (int i = 0; i < ksize; i++)
    {
        for (int j = 0; j < ksize; j++)
        {
            int x = i - (ksize - 1) / 2;
            int y = j - (ksize - 1) / 2;
            kernel[i * ksize + j] = cons * exp(scale * (x * x + y * y));
            sum += kernel[i * ksize + j];
        }
    }
    // normalize
    for (int i = 0; i < ksize; i++)
    {
        for (int j = 0; j < ksize; j++)
        {
            kernel[i * ksize + j] /= sum;
        }
    }
    int kernelSum = sum;

    int width = img.cols;
    int height = img.rows;
    int maskSize = ksize;
    std::vector<uchar> temp_array;
    std::copy(img_array.begin(), img_array.end(), std::back_inserter(temp_array));

    double sum_pix = 0;
    int index = 0;
    int bound = maskSize / 2;
    for (int row = bound; row < height - bound; row++)
    {
        for (int col = bound; col < width - bound; col++)
        {
            index = 0;
            sum_pix = 0;
            for (int i = -bound; i <= bound; i++)
            {
                for (int j = -bound; j <= bound; j++)
                {
                    int pixel_index = (row + i) * width + col + j;
                    if (pixel_index < width * height)
                    {
                        sum_pix += temp_array[pixel_index] * kernel[index];
                        index++;
                    }

                }
            }
            img_array[row * width + col] = uchar(round(sum_pix));
        }
    }

    dst = cv::Mat(img.rows, img.cols, CV_8UC1);
    memcpy(dst.data, img_array.data(), img_array.size() * sizeof(uchar));
    for (int c = 0; c < 10; c++)
    {
        std::cout << (int)dst.at<uchar>(10, c) << std::endl;
    }
}

void Gaussian(const cv::Mat& img, cv::Mat& dst)
{
    cv::GaussianBlur(img, dst, cv::Size(5, 5), 1.0);
    for (int c = 0; c < 10; c++)
    {
        std::cout << (int)dst.at<uchar>(10, c) << std::endl;
    }
}

int main()
{
    cv::Mat img = cv::imread("lena_gray.bmp", cv::IMREAD_GRAYSCALE); // rows512 * cols512
    int channels = img.channels();
    for (int c = 0; c < 10; c++)
    {
        std::cout << (int)img.at<uchar>(10, c) << std::endl;
    }
    std::cout << std::endl;

    cv::Mat dst;
    //Gaussian(img, dst);
    MyGaussian(img, dst);
    cv::imshow("title", dst);
    cv::waitKey(0);
}

/*
Before
156
156
156
160
156
155
155
152
159
159

After
156
156
157
157
157
155
154
155
157
158
*/

/*
https://github.com/egonSchiele/OpenCV/blob/master/modules/imgproc/src/smooth.cpp
https://github.com/egonSchiele/OpenCV/blob/master/modules/imgproc/src/filter.cpp
https://stackoverflow.com/questions/8204645/implementing-gaussian-blur-how-to-calculate-convolution-matrix-kernel/8204880#8204880
https://stackoverflow.com/questions/1696113/how-do-i-gaussian-blur-an-image-without-using-any-in-built-gaussian-functions
https://github.com/ragnraok/android-image-filter/tree/master/library/jni
*/
