package faulttypes;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.util.ArrayUtil;
import org.nd4j.linalg.util.NDArrayUtil;

import utils.ArrayUtilities;

public class FaultFactoryOriginal {
	private int[] label = null;
	private int[] reducedLabel = null;
	private int[][] featureData = null;
	private String type = null;

	private FaultData retFault = null;
	private boolean withNoFault;

	public FaultFactoryOriginal() {
		this(true);
	}

	public FaultFactoryOriginal(boolean withNoFault) {
		this.withNoFault = withNoFault;
	}

	// use getFault method to get object of type Plan
	public FaultData getFault(int type) {
		if (type == 0) {
			this.retFault = new HVPinFault();
		} else if (type == 1) {
			this.retFault = new HVChannelFault();
		} else if (type == 2) {
			this.retFault = new HVConnectorFault();
		} else if (type == 3) {
			this.retFault = new HVFuseFault();
		} else if (type == 4) {
			this.retFault = new HVNoFault();
		} else if (type == 5) {
			this.retFault = new HVDeadWire();
		} else if (type == 6) {
			this.retFault = new HVHotWire();
		}
		makeLabel();
		this.featureData = this.retFault.getData();
		return this.retFault;
	}

	// The array is always made in the following order
	// HVPinFault->HVChannelFault->HVConnectorFault->HVFuseFault->HVDeadWire->HVHotWire

	private void makeLabel() {
		int[] faultArray = this.retFault.getLabel();
		int[] hvPinDeFault = makeFuseDefaultLabel(ArrayUtilities.hvPinFault.length);
		int[] hvChannelDeFault = makeFuseDefaultLabel(ArrayUtilities.hvChannelFault.length);
		int[] hvConnectorDeFault = makeFuseDefaultLabel(ArrayUtilities.hvConnectorFault.length);
		int[] hvFuseDeFault = makeFuseDefaultLabel(ArrayUtilities.hvFuseFault.length);
		int[] hvDeadDeFault = makeFuseDefaultLabel(ArrayUtilities.hvDeadWireFault.length);
		int[] hvHotDeFault = makeFuseDefaultLabel(ArrayUtilities.hvHotWireFault.length);
		int[] hvNoDeFault = makeFuseDefaultLabel(ArrayUtilities.hvNoWireFault.length);

		int[] reducedFaultArray = this.retFault.getReducedLabel();
		int[] hvRePinDeFault = makeFuseDefaultLabel(ArrayUtilities.hvReducedPinFault.length);
		int[] hvReChannelDeFault = makeFuseDefaultLabel(ArrayUtilities.hvReducedChannelFault.length);
		int[] hvReConnectorDeFault = makeFuseDefaultLabel(ArrayUtilities.hvReducedConnectorFault.length);
		int[] hvReFuseDeFault = makeFuseDefaultLabel(ArrayUtilities.hvReducedFuseFault.length);
		int[] hvReDeadDeFault = makeFuseDefaultLabel(ArrayUtilities.hvReducedDeadWireFault.length);
		int[] hvReHotDeFault = makeFuseDefaultLabel(ArrayUtilities.hvReducedHotWireFault.length);
		int[] hvReNoDeFault = makeFuseDefaultLabel(ArrayUtilities.hvReducedNoFault.length);

		if (this.retFault instanceof HVPinFault) {
			hvPinDeFault = faultArray;
			hvRePinDeFault = reducedFaultArray;

		} else if (this.retFault instanceof HVChannelFault) {
			hvChannelDeFault = faultArray;
			hvReChannelDeFault = reducedFaultArray;

		} else if (this.retFault instanceof HVConnectorFault) {
			hvConnectorDeFault = faultArray;
			hvReConnectorDeFault = reducedFaultArray;

		} else if (this.retFault instanceof HVFuseFault) {
			hvFuseDeFault = faultArray;
			hvReFuseDeFault = reducedFaultArray;

		} else if (this.retFault instanceof HVDeadWire) {
			hvDeadDeFault = faultArray;
			hvReDeadDeFault = reducedFaultArray;

		} else if (this.retFault instanceof HVHotWire) {
			hvHotDeFault = faultArray;
			hvReHotDeFault = reducedFaultArray;

		} else if (this.retFault instanceof HVNoFault) {
			hvNoDeFault = faultArray;
			hvReNoDeFault = reducedFaultArray;

		}
		makeLabelArray(hvPinDeFault, hvChannelDeFault, hvConnectorDeFault, hvFuseDeFault, hvDeadDeFault, hvHotDeFault);
		makeReducedLabelArray(hvRePinDeFault, hvReChannelDeFault, hvReConnectorDeFault, hvReFuseDeFault,
				hvReDeadDeFault, hvReHotDeFault);
		this.label = makeLabel(hvPinDeFault, hvChannelDeFault, hvConnectorDeFault, hvFuseDeFault, hvDeadDeFault,
				hvHotDeFault);
		this.reducedLabel = makeLabel(hvRePinDeFault, hvReChannelDeFault, hvReConnectorDeFault, hvReFuseDeFault,
				hvReDeadDeFault, hvReHotDeFault);

	}

	private void makeReducedLabelArray(int[]... coord) {
		this.reducedLabel = new int[getArraySize(coord)];

	}

	private void makeLabelArray(int[]... coord) {
		this.label = new int[getArraySize(coord)];

	}

	private int[] makeLabel(int[]... coord) {
		int[] aLabel = new int[getArraySize(coord)];
		int sizePlacer = 0;
		for (int i = 0; i < coord.length; i++) {
			System.arraycopy(coord[i], 0, aLabel, sizePlacer, coord[i].length);
			sizePlacer += coord[i].length;
		}
		return aLabel;
	}

	private int getArraySize(int[]... coord) {
		int size = 0;
		for (int[] ints : coord) {
			size += ints.length;
		}
		return size;
	}

	private void makeReducedLabelArray(int[] hvRePinDeFault, int[] hvReChannelDeFault, int[] hvReConnectorDeFault,
			int[] hvReFuseDeFault, int[] hvReDeadDeFault, int[] hvReHotDeFault) {
		System.arraycopy(hvRePinDeFault, 0, this.reducedLabel, 0, hvRePinDeFault.length);

		System.arraycopy(hvReChannelDeFault, 0, this.reducedLabel, hvRePinDeFault.length, hvReChannelDeFault.length);

		System.arraycopy(hvReConnectorDeFault, 0, this.reducedLabel, hvRePinDeFault.length + hvReChannelDeFault.length,
				hvReConnectorDeFault.length);

		System.arraycopy(hvReFuseDeFault, 0, this.reducedLabel,
				hvRePinDeFault.length + hvReChannelDeFault.length + hvReConnectorDeFault.length,
				hvReFuseDeFault.length);

		System.arraycopy(hvReDeadDeFault, 0, this.reducedLabel, hvRePinDeFault.length + hvReChannelDeFault.length
				+ hvReConnectorDeFault.length + hvReFuseDeFault.length, hvReDeadDeFault.length);

		System.arraycopy(
				hvReHotDeFault, 0, this.reducedLabel, hvRePinDeFault.length + hvReChannelDeFault.length
						+ hvReConnectorDeFault.length + hvReFuseDeFault.length + hvReDeadDeFault.length,
				hvReHotDeFault.length);
	}

	private void makeLabelArray(int[] hvPinDeFault, int[] hvChannelDeFault, int[] hvConnectorDeFault,
			int[] hvFuseDeFault, int[] hvDeadDeFault, int[] hvHotDeFault) {
		System.arraycopy(hvPinDeFault, 0, this.label, 0, hvPinDeFault.length);

		System.arraycopy(hvChannelDeFault, 0, this.label, hvPinDeFault.length, hvChannelDeFault.length);

		System.arraycopy(hvConnectorDeFault, 0, this.label, hvPinDeFault.length + hvChannelDeFault.length,
				hvConnectorDeFault.length);

		System.arraycopy(hvFuseDeFault, 0, this.label,
				hvPinDeFault.length + hvChannelDeFault.length + hvConnectorDeFault.length, hvFuseDeFault.length);

		System.arraycopy(hvDeadDeFault, 0, this.label,
				hvPinDeFault.length + hvChannelDeFault.length + hvConnectorDeFault.length + hvFuseDeFault.length,
				hvDeadDeFault.length);

		System.arraycopy(hvHotDeFault, 0, this.label, hvPinDeFault.length + hvChannelDeFault.length
				+ hvConnectorDeFault.length + hvFuseDeFault.length + hvDeadDeFault.length, hvHotDeFault.length);
	}

	private int[] makeFuseDefaultLabel(int length) {
		int[] deFault = new int[length];
		for (int i = 0; i < deFault.length; i++) {
			deFault[i] = 0;
		}
		return deFault;
	}

	public INDArray getLabelVector() {
		return NDArrayUtil.toNDArray(this.label);
	}

	public INDArray getFeatureVector() {
		return NDArrayUtil.toNDArray(ArrayUtil.flatten(this.featureData));
	}

	public int[] getFeatureArray() {
		return ArrayUtil.flatten(this.featureData);
	}

	public int[] getLabel() {
		return label;
	}

	public int[] getReducedLabel() {
		return reducedLabel;
	}

	public int[][] getFeatureData() {
		return featureData;
	}

	public String getType() {
		return type;
	}
}// end of FaultFactory class.
