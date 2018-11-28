package client;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Scanner;

import org.nd4j.linalg.api.ndarray.INDArray;

import clasDC.faults.FaultNames;
import processHipo.DataProcess;
import strategies.FaultRecordScalerStrategy;
import strategies.MinMaxStrategy;

public class ValidateRealData {

	public static void main(String[] args) throws IOException {

		// String dataDir = "data/";
		String dataDir = "/Volumes/MacStorage/WorkData/CLAS12/RGACooked/V5b.2.1/";// DomainUtils.getDataLocation();

		List<String> aList = new ArrayList<>();
		aList.add(dataDir + "out_clas_003923.evio.80.hipo");
		aList.add(dataDir + "out_clas_003923.evio.8.hipo");

		DataProcess dataProcess = new DataProcess(aList);
		dataProcess.processFile();

		FaultRecordScalerStrategy strategy = new MinMaxStrategy();

		for (int sector = 1; sector < 7; sector++) {
			for (int superlayer = 1; superlayer < 7; superlayer++) {
				double smallPinCertainty = getCertainty(FaultNames.PIN_SMALL, superlayer,
						dataProcess.getFeatureVector(sector, superlayer, strategy));
				double bigPinCertainty = getCertainty(FaultNames.PIN_BIG, superlayer,
						dataProcess.getFeatureVector(sector, superlayer, strategy));
				double deadWireCertainty = getCertainty(FaultNames.DEADWIRE, superlayer,
						dataProcess.getFeatureVector(sector, superlayer, strategy));
				double hotWireCertainty = getCertainty(FaultNames.HOTWIRE, superlayer,
						dataProcess.getFeatureVector(sector, superlayer, strategy));
				double connectorECertainty = getCertainty(FaultNames.CONNECTOR_E, superlayer,
						dataProcess.getFeatureVector(sector, superlayer, strategy));
				double connectorTreeCertainty = getCertainty(FaultNames.CONNECTOR_TREE, superlayer,
						dataProcess.getFeatureVector(sector, superlayer, strategy));
				double connectorThreeCertainty = getCertainty(FaultNames.CONNECTOR_THREE, superlayer,
						dataProcess.getFeatureVector(sector, superlayer, strategy));
				double fuseACertainty = getCertainty(FaultNames.FUSE_A, superlayer,
						dataProcess.getFeatureVector(sector, superlayer, strategy));
				double fuseBCertainty = getCertainty(FaultNames.FUSE_B, superlayer,
						dataProcess.getFeatureVector(sector, superlayer, strategy));
				double fuseCCertainty = getCertainty(FaultNames.FUSE_C, superlayer,
						dataProcess.getFeatureVector(sector, superlayer, strategy));
				double channelOneCertainty = getCertainty(FaultNames.CHANNEL_ONE, superlayer,
						dataProcess.getFeatureVector(sector, superlayer, strategy));
				double channelTwoCertainty = getCertainty(FaultNames.CHANNEL_TWO, superlayer,
						dataProcess.getFeatureVector(sector, superlayer, strategy));
				double channelThreeCertainty = getCertainty(FaultNames.CHANNEL_THREE, superlayer,
						dataProcess.getFeatureVector(sector, superlayer, strategy));

				dataProcess.plotData(sector, superlayer);

				boolean printAll = false;

				System.out.println("Detected Faults:");
				if (smallPinCertainty > 0.5 || printAll) {
					System.out.println("Small Pin: " + smallPinCertainty * 100 + "%");
				}
				if (bigPinCertainty > 0.5 || printAll) {
					System.out.println("Big Pin: " + bigPinCertainty * 100 + "%");
				}
				if (deadWireCertainty > 0.5 || printAll) {
					System.out.println("Dead Wire: " + deadWireCertainty * 100 + "%");
				}
				if (hotWireCertainty > 0.5 || printAll) {
					System.out.println("Hot Wire: " + hotWireCertainty * 100 + "%");
				}
				if (connectorECertainty > 0.5 || printAll) {
					System.out.println("Connector E: " + connectorECertainty * 100 + "%");
				}
				if (connectorTreeCertainty > 0.5 || printAll) {
					System.out.println("Connector Tree: " + connectorTreeCertainty * 100 + "%");
				}
				if (connectorThreeCertainty > 0.5 || printAll) {
					System.out.println("Connector Three: " + connectorThreeCertainty * 100 + "%");
				}
				if (fuseACertainty > 0.5 || printAll) {
					System.out.println("Fuse A: " + fuseACertainty * 100 + "%");
				}
				if (fuseBCertainty > 0.5 || printAll) {
					System.out.println("Fuse B: " + fuseBCertainty * 100 + "%");
				}
				if (fuseCCertainty > 0.5 || printAll) {
					System.out.println("Fuse C: " + fuseCCertainty * 100 + "%");
				}
				if (channelOneCertainty > 0.5 || printAll) {
					System.out.println("Channel One: " + channelOneCertainty * 100 + "%");
				}
				if (channelTwoCertainty > 0.5 || printAll) {
					System.out.println("Channel Two: " + channelTwoCertainty * 100 + "%");
				}
				if (channelThreeCertainty > 0.5 || printAll) {
					System.out.println("Channel Three: " + channelThreeCertainty * 100 + "%");
				}

				Scanner sc = new Scanner(System.in);
				sc.nextLine();
			}
		}

	}

	public static double getCertainty(FaultNames fault, int superlayer, INDArray data) throws IOException {
		// get the model
		FaultClassifier classifier = new FaultClassifier(
				"models/binary_classifiers/SL" + superlayer + "/" + fault + ".zip");
		double[] predictions = classifier.output(data).toDoubleVector();
		return predictions[0];
	}
}
