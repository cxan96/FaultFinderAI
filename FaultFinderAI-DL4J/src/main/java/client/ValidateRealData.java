package client;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Scanner;

import org.nd4j.linalg.api.ndarray.INDArray;

import faults.FaultNames;
import processHipo.DataProcess;
import strategies.*;

public class ValidateRealData {

    public static void main(String[] args) throws IOException{

	String dataDir = "data/";
		    
	List<String> aList = new ArrayList<>();
	aList.add(dataDir + "out_clas_003923.evio.80.hipo");
	aList.add(dataDir + "out_clas_003923.evio.8.hipo");
		
	DataProcess dataProcess = new DataProcess(aList);
	dataProcess.processFile();

	FaultRecordScalerStrategy strategy = new MinMaxStrategy();

	for (int sector = 1; sector < 2; sector++) {
	    for (int superlayer = 1; superlayer < 7; superlayer++) {
		double smallPinCertainty = getCertainty(FaultNames.PIN_SMALL, sector, dataProcess.getFeatureVector(sector, superlayer, strategy));
		double bigPinCertainty = getCertainty(FaultNames.PIN_BIG, sector, dataProcess.getFeatureVector(sector, superlayer, strategy));
		double deadWireCertainty = getCertainty(FaultNames.DEADWIRE, sector, dataProcess.getFeatureVector(sector, superlayer, strategy));
		double hotWireCertainty = getCertainty(FaultNames.HOTWIRE, sector, dataProcess.getFeatureVector(sector, superlayer, strategy));

		dataProcess.plotData(sector, superlayer);
		
		System.out.println("Small Pin: "+smallPinCertainty*100+"%");
		System.out.println("Big Pin: "+bigPinCertainty*100+"%");
		System.out.println("Dead Wire: "+deadWireCertainty*100+"%");
		System.out.println("Hot Wire: "+hotWireCertainty*100+"%");

		Scanner sc = new Scanner(System.in);
		sc.nextLine();
	    }
	}

    }

    public static double getCertainty(FaultNames fault, int sector, INDArray data) throws IOException{
	// get the model
	FaultClassifier classifier = new FaultClassifier("models/binary_classifiers/sector"+sector+"/"+fault+".zip");
	double[] predictions = classifier.output(data).toDoubleVector();
	return predictions[0];
    }
}
