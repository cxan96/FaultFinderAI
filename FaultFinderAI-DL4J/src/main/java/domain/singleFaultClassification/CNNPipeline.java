package domain.singleFaultClassification;

import java.io.IOException;

import org.nd4j.linalg.api.ndarray.INDArray;

import client.FaultClassifier;

/**
 * Used for the One vs. All CNN arcitecture
 * 
 * @author michaelkunkel
 *
 */
public class CNNPipeline {

	private String path = "models/";
	private String HVPinFaultFile = "SingleFaultClassifyingHVPinFault.zip";
	private FaultClassifier HVPinFaultClassifier = null;
	private FaultClassifier HVChannelFaultClassifier = null;
	private FaultClassifier HVConnectorFaultClassifier = null;
	private FaultClassifier HVFuseFaultClassifier = null;
	private FaultClassifier HVDeadWireClassifier = null;
	private FaultClassifier HVHotWireClassifier = null;

	public CNNPipeline() {
		initClassifiers();
	}

	public static void main(String[] args) {
	}

	private void initClassifiers() {
		try {
			HVPinFaultClassifier = new FaultClassifier(path + HVPinFaultFile);
			/**
			 * Add more here
			 */
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}

	public static void pipe(INDArray array) {

	}

}
