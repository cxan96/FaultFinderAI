package faulttypes;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.util.ArrayUtil;
import org.nd4j.linalg.util.NDArrayUtil;

import utils.ArrayUtilities;

public class FaultFactory {
	private int[] label = null;
	private int[] reducedLabel = null;
	private int[][] featureData = null;
	private String type = null;

	private FaultData retFault = null;
	private boolean withHotwireFault;

	public FaultFactory() {
		this(true);
	}

	public FaultFactory(boolean withHotwireFault) {
		this.withHotwireFault = withHotwireFault;
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
	public void plotData() {
		this.retFault.plotData();
	}

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
		if (withHotwireFault) {
			this.label = makeLabel(hvPinDeFault, hvChannelDeFault, hvConnectorDeFault, hvFuseDeFault, hvDeadDeFault,
					hvHotDeFault, hvNoDeFault);
			this.reducedLabel = makeLabel(hvRePinDeFault, hvReChannelDeFault, hvReConnectorDeFault, hvReFuseDeFault,
					hvReDeadDeFault, hvReHotDeFault, hvReNoDeFault);
		} else {
			this.label = makeLabel(hvPinDeFault, hvChannelDeFault, hvConnectorDeFault, hvFuseDeFault, hvDeadDeFault,
					hvNoDeFault);
			this.reducedLabel = makeLabel(hvRePinDeFault, hvReChannelDeFault, hvReConnectorDeFault, hvReFuseDeFault,
					hvReDeadDeFault, hvReNoDeFault);
		}

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
		return this.label;
	}

	/**
	 * 
	 * getFaultLabel(): The label for an individual fault i.e. HVChannel will
	 * have int[8]
	 * 
	 */

	public int[] getFaultLabel() {
		return this.retFault.getLabel();
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

	public int getReducedFaultIndex() {
		int retVal = 0;
		for (int i = 0; i < this.reducedLabel.length; i++) {
			if (this.reducedLabel[i] == 1) {
				retVal = i;
			}
		}
		return retVal;
	}

	public String getFaultName() {
		return this.retFault.getClass().getSimpleName();
	}
}// end
	// of
	// FaultFactory
	// class.
