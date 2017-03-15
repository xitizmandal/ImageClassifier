package test;

import org.opencv.core.Core;
import org.opencv.core.Mat;

import classifiers.OneVsAllClassifier;
import classifiers.OneVsOneClassifier;
import utils.Utils;

public class Testing {	
	public Testing(){
		System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
	}
	
	public void test(String filename, int sampleNo) {
		OneVsAllClassifier oneVsAllClassifier = new OneVsAllClassifier(sampleNo);
		Mat out = new Mat();
		out = Utils.loadAndResizeImage(Utils.TEST_PATH + filename);
		
		System.out.println(Utils.classification[sampleNo] + ":\t" + oneVsAllClassifier.getClassifier().predict(out));
		System.out.println(oneVsAllClassifier.getClassifier().predict(out,true));
	}
	
	public void test(String filename) {
		OneVsOneClassifier oneVsOneClassifier = new OneVsOneClassifier();
	    Mat out = new Mat();
	    out = Utils.loadAndResizeImage(Utils.TEST_PATH+filename);
	    
	    System.out.println(Utils.returnValString(oneVsOneClassifier.getClassifier().predict(out)));
	    System.out.println(oneVsOneClassifier.getClassifier().predict(out,true));
		}
}
