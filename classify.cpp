#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/dnn.hpp>
#include "opencv2/imgproc/imgproc_c.h"

#include <iostream>
#include <fstream>

using namespace std;
using namespace cv;
using namespace dnn;



int main(int argc, char** argv)
{   
  
  // Load the network
  Net net = readNetFromONNX("resnet18.onnx");
  net.setPreferableBackend(DNN_BACKEND_OPENCV);
  net.setPreferableTarget(DNN_TARGET_CPU);

  // get the network outputs

  vector<string> classes;
  const string file = "synset.txt";
  ifstream ifs(file.c_str());
  if (!ifs.is_open()){
    cout << "fuck off." << endl;
    return -1;
  }
  string line;
  while (getline(ifs, line)){
    classes.push_back(line);
  }



  // Setup IP cam
  string ip ;

  ifstream get_ip("ip", ios::in); 

  if(get_ip){  
         

    getline(get_ip, ip);  
    cout << ip << endl;  
    get_ip.close();

  } 
  //cout << "Please enter your cam IP" << endl << endl ;
  //cin >> ip;
  VideoCapture cam;
  cam.open(ip);

  if(!cam.isOpened()){

    cout << "the cam failed to open" << endl;
    return -1;
  }
  Mat frame, blob;

  // Read a frame
  for(;;){
    cam >> frame ;

    // Create a 4D blob from a frame for resnet
    blobFromImage(frame, blob, 1/255.0, cvSize(224, 224), Scalar(0,0,0), true, false);

    //Sets the input to the network
    net.setInput(blob);
    Mat prob = net.forward();

    Point classPoint;
    double confidence;
    minMaxLoc(prob.reshape(1, 1), 0, &confidence, 0, &classPoint);
    int id = classPoint.x;

    // Print predicted class.
    string label = format("%s: %.4f", (classes.empty() ? format("Class #%d", id).c_str() : classes[id].c_str()), confidence);
    putText(frame, label, Point(0, 40), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 0));

    imshow("I see you", frame);

    if (waitKey(2) >= 0) break;

  }
  cam.release();
  return 0;
}
