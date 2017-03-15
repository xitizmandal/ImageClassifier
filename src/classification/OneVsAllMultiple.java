package classification;

import org.opencv.core.Core;

import classifiers.OneVsAllClassifier;
import utils.Utils;

public class OneVsAllMultiple {	
	double lowestVal = 10;
	float classifierVal;
	int index;
	
	public OneVsAllMultiple(){
		System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
	}
	public void startDetection(String filename){
		for (int i = 0; i < Utils.NUMBER_OF_CLASSIFICATION;i++){
			OneVsAllClassifier svm = new OneVsAllClassifier(i);
			classifierVal = svm.getClassifier().predict(Utils.loadAndResizeImage(Utils.TEST_PATH+filename),true);
			System.out.println(Utils.classification[i] + ":\t" + classifierVal);
			
			if (classifierVal < lowestVal){
				lowestVal = classifierVal;
				index = i;
			}
			svm.close();
		}
		System.out.println(filename + " ----> " + Utils.classification[index]);
	}
}
