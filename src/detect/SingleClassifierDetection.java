package detect;

import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.highgui.Highgui;

import utils.Pyramid;
import utils.SlidingWindow;
import utils.Utils;

public class SingleClassifierDetection {


	int index;
	int windowSizeC;
	int windowSizeR;
	double windowSizeIncrement;
	int stepSize;
	Mat inputImage;
	
	public SingleClassifierDetection(int sampleNo, int windowSizeC, int windowSizeR, double windowSizeIncrement, int stepSize){
		this.windowSizeC = windowSizeC;
		this.windowSizeR = windowSizeR;
		this.windowSizeIncrement = windowSizeIncrement;
		this.stepSize = stepSize;
		this.index = sampleNo;
		
		System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
	}
	
	public SingleClassifierDetection(int sampleNo){
		this(sampleNo, 128,128, 1.25,8);
	}
	
	public void startDetection(String filename){
			inputImage = Highgui.imread(Utils.TEST_PATH+filename);
			Pyramid pyramid = new Pyramid(inputImage, windowSizeC, windowSizeR, windowSizeIncrement, stepSize);
			
			while(true){
				int terminate = 0;
				
				//System.out.println(pyramid.getColumnWindowSize() +","+ pyramid.getRowWindowSize() + "," + inputImage.cols() + "," + inputImage.rows());
				SlidingWindow slidingWindow = new SlidingWindow(inputImage, pyramid.getColumnWindowSize(),
						pyramid.getRowWindowSize(), stepSize, index);
				slidingWindow.startWindow();
				
				if ((inputImage.cols() - pyramid.getTestColumnWindowSize() - stepSize) > 0){
					pyramid.setColumnWindowSize(pyramid.getTestColumnWindowSize());
					terminate++;
				}
				
				if ((inputImage.rows() - pyramid.getTestRowWindowSize() - stepSize) > 0){
					pyramid.setRowWindowSize(pyramid.getTestRowWindowSize());
					terminate++;
				}
				
				if (terminate == 0){
					break;
				}
				
			}
	}
}
