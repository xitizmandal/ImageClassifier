package utils;

import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.highgui.Highgui;

import classifiers.OneVsAllClassifier;

public class SlidingWindow {
	int windowSizeC;
	int windowSizeR;
	int stepSize;
	int index;
	Mat image;
	OneVsAllClassifier svm;
	
	public SlidingWindow (Mat inputImage, int windowSizeC, int windowSizeR,int stepSize, int index){
		this.image = inputImage;
		this.windowSizeC = windowSizeC;
		this.windowSizeR = windowSizeR;
		this.stepSize = stepSize;
		
		this.index = index;
		
		svm = new OneVsAllClassifier(index);
				
//		startWindow();
	}
	
	public void setParams(int windowSizeC, int windowSizeR, int stepSize){
		this.windowSizeC = windowSizeC;
		this.windowSizeR = windowSizeR;
		this.stepSize = stepSize;
	}
	
	public void startWindow(){				
		for (int cols = 0;cols <= (image.cols()- windowSizeC); cols += stepSize){
			for (int rows = 0;rows <= (image.rows()- windowSizeR); rows += stepSize) {
				Rect roi = new Rect(cols,rows,windowSizeC,windowSizeR);
				float classifierVal = svm.getClassifier().predict(Utils.loadAndResizeImage(image.submat(roi)),true);
				
				if (classifierVal < 0){
					Mat outputImage = image.clone();
					
					Core.rectangle(outputImage, new Point(cols, rows), new Point(cols + windowSizeC, rows + windowSizeR),
		                    new Scalar(0, 0, 255));
					
					String fileName = Utils.OUTPUT_PATH +Utils.classification[index] + "_"+ cols + "_" + rows+ "_"
							+ windowSizeC + "_" + windowSizeR +".png";
					
					Highgui.imwrite(fileName, outputImage);	
					
					System.out.println(fileName);
				}
			}
		}
	}

}
