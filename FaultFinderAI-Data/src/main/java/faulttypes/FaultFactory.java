package faulttypes;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.util.ArrayUtil;
import org.nd4j.linalg.util.NDArrayUtil;

import utils.ArrayUtilities;

public class FaultFactory {
	private int[] faultLabel = null;
	int[][] featureData = null;

	public FaultFactory() {
		this.faultLabel = new int[ArrayUtilities.faultLableSize];
	}

	// use getPlan method to get object of type Plan
	public FaultData getFault(int type) {
		FaultData retFault = null;
		if (type == 0) {
			retFault = new HVPinFault();
		} else if (type == 1) {
			retFault = new HVChannelFault();
		} else if (type == 2) {
			retFault = new HVConnectorFault();
		} else if (type == 3) {
			retFault = new HVFuseFault();
		} else if (type == 4) {
			retFault = new HVDeadWire();
		} else if (type == 5) {
			retFault = new HVHotWire();
		}
		makeLabel(retFault);
		this.featureData = retFault.getData();
		return retFault;
	}

	public int[] getLabel() {
		return this.faultLabel;
	}

	// The array is always made in the following order
	// HVPinFault->HVChannelFault->HVConnectorFault->HVFuseFault->HVDeadWire->HVHotWire

	private void makeLabel(FaultData fault) {
		int[] faultArray = fault.getFaultLabel();
		int[] hvPinDeFault = makeFuseDefaultLabel(ArrayUtilities.hvPinFault.length);
		int[] hvChannelDeFault = makeFuseDefaultLabel(ArrayUtilities.hvChannelFault.length);
		int[] hvConnectorDeFault = makeFuseDefaultLabel(ArrayUtilities.hvConnectorFault.length);
		int[] hvFuseDeFault = makeFuseDefaultLabel(ArrayUtilities.hvFuseFault.length);
		int[] hvDeadDeFault = makeFuseDefaultLabel(ArrayUtilities.hvWireFault.length);
		int[] hvHotDeFault = makeFuseDefaultLabel(ArrayUtilities.hvWireFault.length);
		if (fault instanceof HVPinFault) {
			hvPinDeFault = faultArray;
			System.arraycopy(hvPinDeFault, 0, this.faultLabel, 0, hvPinDeFault.length);

			System.arraycopy(hvChannelDeFault, 0, this.faultLabel, hvPinDeFault.length, hvChannelDeFault.length);

			System.arraycopy(hvConnectorDeFault, 0, this.faultLabel, hvPinDeFault.length + hvChannelDeFault.length,
					hvConnectorDeFault.length);

			System.arraycopy(hvFuseDeFault, 0, this.faultLabel,
					hvPinDeFault.length + hvChannelDeFault.length + hvConnectorDeFault.length, hvFuseDeFault.length);

			System.arraycopy(hvDeadDeFault, 0, this.faultLabel,
					hvPinDeFault.length + hvChannelDeFault.length + hvConnectorDeFault.length + hvFuseDeFault.length,
					hvDeadDeFault.length);

			System.arraycopy(
					hvHotDeFault, 0, this.faultLabel, hvPinDeFault.length + hvChannelDeFault.length
							+ hvConnectorDeFault.length + hvFuseDeFault.length + hvDeadDeFault.length,
					hvHotDeFault.length);

		} else if (fault instanceof HVChannelFault) {
			hvChannelDeFault = faultArray;
			System.arraycopy(hvPinDeFault, 0, this.faultLabel, 0, hvPinDeFault.length);

			System.arraycopy(hvChannelDeFault, 0, this.faultLabel, hvPinDeFault.length, hvChannelDeFault.length);

			System.arraycopy(hvConnectorDeFault, 0, this.faultLabel, hvPinDeFault.length + hvChannelDeFault.length,
					hvConnectorDeFault.length);

			System.arraycopy(hvFuseDeFault, 0, this.faultLabel,
					hvPinDeFault.length + hvChannelDeFault.length + hvConnectorDeFault.length, hvFuseDeFault.length);

			System.arraycopy(hvDeadDeFault, 0, this.faultLabel,
					hvPinDeFault.length + hvChannelDeFault.length + hvConnectorDeFault.length + hvFuseDeFault.length,
					hvDeadDeFault.length);

			System.arraycopy(
					hvHotDeFault, 0, this.faultLabel, hvPinDeFault.length + hvChannelDeFault.length
							+ hvConnectorDeFault.length + hvFuseDeFault.length + hvDeadDeFault.length,
					hvHotDeFault.length);

		} else if (fault instanceof HVConnectorFault) {
			hvConnectorDeFault = faultArray;
			System.arraycopy(hvPinDeFault, 0, this.faultLabel, 0, hvPinDeFault.length);

			System.arraycopy(hvChannelDeFault, 0, this.faultLabel, hvPinDeFault.length, hvChannelDeFault.length);

			System.arraycopy(hvConnectorDeFault, 0, this.faultLabel, hvPinDeFault.length + hvChannelDeFault.length,
					hvConnectorDeFault.length);

			System.arraycopy(hvFuseDeFault, 0, this.faultLabel,
					hvPinDeFault.length + hvChannelDeFault.length + hvConnectorDeFault.length, hvFuseDeFault.length);

			System.arraycopy(hvDeadDeFault, 0, this.faultLabel,
					hvPinDeFault.length + hvChannelDeFault.length + hvConnectorDeFault.length + hvFuseDeFault.length,
					hvDeadDeFault.length);

			System.arraycopy(
					hvHotDeFault, 0, this.faultLabel, hvPinDeFault.length + hvChannelDeFault.length
							+ hvConnectorDeFault.length + hvFuseDeFault.length + hvDeadDeFault.length,
					hvHotDeFault.length);

		} else if (fault instanceof HVFuseFault) {
			hvFuseDeFault = faultArray;
			System.arraycopy(hvPinDeFault, 0, this.faultLabel, 0, hvPinDeFault.length);

			System.arraycopy(hvChannelDeFault, 0, this.faultLabel, hvPinDeFault.length, hvChannelDeFault.length);

			System.arraycopy(hvConnectorDeFault, 0, this.faultLabel, hvPinDeFault.length + hvChannelDeFault.length,
					hvConnectorDeFault.length);

			System.arraycopy(hvFuseDeFault, 0, this.faultLabel,
					hvPinDeFault.length + hvChannelDeFault.length + hvConnectorDeFault.length, hvFuseDeFault.length);

			System.arraycopy(hvDeadDeFault, 0, this.faultLabel,
					hvPinDeFault.length + hvChannelDeFault.length + hvConnectorDeFault.length + hvFuseDeFault.length,
					hvDeadDeFault.length);

			System.arraycopy(
					hvHotDeFault, 0, this.faultLabel, hvPinDeFault.length + hvChannelDeFault.length
							+ hvConnectorDeFault.length + hvFuseDeFault.length + hvDeadDeFault.length,
					hvHotDeFault.length);

		} else if (fault instanceof HVDeadWire) {
			hvDeadDeFault = faultArray;
			System.arraycopy(hvPinDeFault, 0, this.faultLabel, 0, hvPinDeFault.length);

			System.arraycopy(hvChannelDeFault, 0, this.faultLabel, hvPinDeFault.length, hvChannelDeFault.length);

			System.arraycopy(hvConnectorDeFault, 0, this.faultLabel, hvPinDeFault.length + hvChannelDeFault.length,
					hvConnectorDeFault.length);

			System.arraycopy(hvFuseDeFault, 0, this.faultLabel,
					hvPinDeFault.length + hvChannelDeFault.length + hvConnectorDeFault.length, hvFuseDeFault.length);

			System.arraycopy(hvDeadDeFault, 0, this.faultLabel,
					hvPinDeFault.length + hvChannelDeFault.length + hvConnectorDeFault.length + hvFuseDeFault.length,
					hvDeadDeFault.length);

			System.arraycopy(
					hvHotDeFault, 0, this.faultLabel, hvPinDeFault.length + hvChannelDeFault.length
							+ hvConnectorDeFault.length + hvFuseDeFault.length + hvDeadDeFault.length,
					hvHotDeFault.length);

		} else if (fault instanceof HVHotWire) {
			hvHotDeFault = faultArray;
			System.arraycopy(hvPinDeFault, 0, this.faultLabel, 0, hvPinDeFault.length);

			System.arraycopy(hvChannelDeFault, 0, this.faultLabel, hvPinDeFault.length, hvChannelDeFault.length);

			System.arraycopy(hvConnectorDeFault, 0, this.faultLabel, hvPinDeFault.length + hvChannelDeFault.length,
					hvConnectorDeFault.length);

			System.arraycopy(hvFuseDeFault, 0, this.faultLabel,
					hvPinDeFault.length + hvChannelDeFault.length + hvConnectorDeFault.length, hvFuseDeFault.length);

			System.arraycopy(hvDeadDeFault, 0, this.faultLabel,
					hvPinDeFault.length + hvChannelDeFault.length + hvConnectorDeFault.length + hvFuseDeFault.length,
					hvDeadDeFault.length);

			System.arraycopy(
					hvHotDeFault, 0, this.faultLabel, hvPinDeFault.length + hvChannelDeFault.length
							+ hvConnectorDeFault.length + hvFuseDeFault.length + hvDeadDeFault.length,
					hvHotDeFault.length);

		}
	}

	private int[] makeFuseDefaultLabel(int length) {
		int[] deFault = new int[length];
		for (int i = 0; i < deFault.length; i++) {
			deFault[i] = 0;
		}
		return deFault;
	}

	public INDArray getLabelVector() {
		return NDArrayUtil.toNDArray(this.faultLabel);
	}

	public INDArray getFeatureVector() {
		return NDArrayUtil.toNDArray(ArrayUtil.flatten(this.featureData));
	}

	public int[] getLabelArray() {
		return this.faultLabel;
	}

	public int[] getFeatureArray() {
		return ArrayUtil.flatten(this.featureData);
	}
}// end of FaultFactory class.
