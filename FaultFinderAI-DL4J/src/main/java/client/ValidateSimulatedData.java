package client;

import java.io.IOException;
import java.util.Scanner;

import org.nd4j.linalg.api.ndarray.INDArray;

import clasDC.factories.FaultFactory;
import clasDC.faults.FaultNames;
import strategies.FaultRecordScalerStrategy;
import strategies.MinMaxStrategy;

public class ValidateSimulatedData {

	public static void main(String[] args) throws IOException {

		FaultRecordScalerStrategy strategy = new MinMaxStrategy();
		int maxFaults = 5;

		for (int superlayer = 1; superlayer <= 6; superlayer++) {
			for (int i = 0; i < 5; i++) {

				FaultFactory factory = new FaultFactory(superlayer, maxFaults, FaultNames.DEADWIRE, false);
				factory.draw();
				INDArray faultData = factory.getFeatureVector();
				strategy.normalize(faultData);

				double smallPinCertainty = getCertainty(FaultNames.PIN_SMALL, superlayer, faultData);
				double bigPinCertainty = getCertainty(FaultNames.PIN_BIG, superlayer, faultData);
				double deadWireCertainty = getCertainty(FaultNames.DEADWIRE, superlayer, faultData);
				double hotWireCertainty = getCertainty(FaultNames.HOTWIRE, superlayer, faultData);
				double connectorECertainty = getCertainty(FaultNames.CONNECTOR_E, superlayer, faultData);
				double connectorTreeCertainty = getCertainty(FaultNames.CONNECTOR_TREE, superlayer, faultData);
				double connectorThreeCertainty = getCertainty(FaultNames.CONNECTOR_THREE, superlayer, faultData);
				double fuseACertainty = getCertainty(FaultNames.FUSE_A, superlayer, faultData);
				double fuseBCertainty = getCertainty(FaultNames.FUSE_B, superlayer, faultData);
				double fuseCCertainty = getCertainty(FaultNames.FUSE_C, superlayer, faultData);
				double channelOneCertainty = getCertainty(FaultNames.CHANNEL_ONE, superlayer, faultData);
				double channelTwoCertainty = getCertainty(FaultNames.CHANNEL_TWO, superlayer, faultData);
				double channelThreeCertainty = getCertainty(FaultNames.CHANNEL_THREE, superlayer, faultData);

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
