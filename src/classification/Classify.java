package classification;

import detect.SingleClassifierDetection;
import utils.Utils;

public class Classify {
	public static void main(String[] args){
		Utils.checkAllDirectories();
		String inputFile = "panda3.jpeg";
		int sampleNo = 1;
		
		OneVsAll oneVsAll = new OneVsAll();
		oneVsAll.test(inputFile, sampleNo);
				
		OneVsOne oneVsOne = new OneVsOne();
//		oneVsOne.train();
		oneVsOne.test(inputFile);
		
		OneVsAllMultiple oneVsAllMultiple = new OneVsAllMultiple();
		oneVsAllMultiple.startDetection(inputFile);
		
		SingleClassifierDetection singleClassifierDetection = new SingleClassifierDetection(sampleNo);
		singleClassifierDetection.startDetection(inputFile);
		
		
	}
}

