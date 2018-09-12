package client;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import org.nd4j.linalg.api.ndarray.INDArray;

import faults.FaultNames;
import processHipo.DataProcess;
import strategies.FaultRecordScalerStrategy;
import strategies.MinMaxStrategy;

public class ClassifyRealData {

	private String dataDir = null;
	private List<String> aList = null;
	private List<FaultNames> fautList = null;

	private DataProcess dataProcess;
	private FaultRecordScalerStrategy strategy;
	private boolean singleModels = false;

	public ClassifyRealData() {
		this.dataDir = "/Volumes/MacStorage/WorkData/CLAS12/RGACooked/V5b.2.1/";
		// this.dataDir =
		// "/Users/michaelkunkel/WORK/CLAS/CLAS12/CLAS12Data/RGACooked/V5b.2.1/";
		this.aList = new ArrayList<>();
		fautList = new ArrayList<>();

		makeList();
		this.dataProcess = new DataProcess(aList);
		this.strategy = new MinMaxStrategy();
	}

	private void makeList() {
		aList.add(dataDir + "out_clas_003923.evio.80.hipo");
		// aList.add(dataDir + "out_clas_003923.evio.8.hipo");

		// aList.add(dataDir + "out_clas_003971.evio.1000.hipo");
		// aList.add(dataDir + "out_clas_003971.evio.1001.hipo");
		// aList.add(dataDir + "out_clas_003971.evio.1002.hipo");
		// aList.add(dataDir + "out_clas_003971.evio.1003.hipo");
		// aList.add(dataDir + "out_clas_003971.evio.1004.hipo");

		fautList.add(FaultNames.CHANNEL_ONE);
		fautList.add(FaultNames.CHANNEL_TWO);
		fautList.add(FaultNames.CHANNEL_THREE);

		fautList.add(FaultNames.CONNECTOR_E);
		fautList.add(FaultNames.CONNECTOR_THREE);
		fautList.add(FaultNames.CONNECTOR_TREE);

		fautList.add(FaultNames.FUSE_A);
		fautList.add(FaultNames.FUSE_B);
		fautList.add(FaultNames.FUSE_C);

		fautList.add(FaultNames.DEADWIRE);

		fautList.add(FaultNames.HOTWIRE);

		fautList.add(FaultNames.PIN_BIG);
		fautList.add(FaultNames.PIN_SMALL);
	}

	public void runSingleModels() throws IOException {
		dataProcess.processFile();
		for (int sector = 1; sector < 7; sector++) {
			for (int superlayer = 1; superlayer < 7; superlayer++) {
				dataProcess.plotData(sector, superlayer);
				System.out.println("\nDetected Faults for Sector: " + sector + " SuperLayer: " + superlayer);
				INDArray featureArray = dataProcess.getFeatureVector(sector, superlayer, strategy);
				for (FaultNames fault : fautList) {
					printCertainty(fault, superlayer, featureArray, false);
				}
			}
		}
	}

	public double getCertainty(FaultNames fault, int superlayer, INDArray data) throws IOException {
		// get the model
		FaultClassifier classifier;
		if (!singleModels) {
			classifier = new FaultClassifier(
					"models/binary_classifiers/SL" + superlayer + "/" + fault.getSaveName() + ".zip");
		} else {
			classifier = new FaultClassifier(
					"models/binary_classifiers/benchmark/" + fault.getSaveName() + "_save1.zip");
		}
		double[] predictions = classifier.output(data).toDoubleVector();
		return predictions[0];
	}

	public void printCertainty(FaultNames fault, int superlayer, INDArray data, boolean printAll) throws IOException {
		// get the model
		FaultClassifier classifier;
		if (!singleModels) {
			classifier = new FaultClassifier(
					"models/binary_classifiers/SL" + superlayer + "/" + fault.getSaveName() + ".zip");
		} else {
			classifier = new FaultClassifier(
					"models/binary_classifiers/SmearedFaults/" + fault.getSaveName() + "_save1.zip");
		}
		double[] predictions = classifier.output(data).toDoubleVector();
		if (printAll) {
			System.out.println(fault + "  " + predictions[0] * 100);
		}
		if (predictions[0] > 0.5 && !printAll) {
			System.out.println(fault + "  " + predictions[0] * 100);
		}

	}

	public void setSingleModel(boolean singleModels) {
		this.singleModels = singleModels;
	}

	public static void main(String[] args) throws IOException {
		ClassifyRealData cData = new ClassifyRealData();
		cData.setSingleModel(true);
		cData.runSingleModels();
	}

}
