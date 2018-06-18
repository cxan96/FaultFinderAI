package faulttypes;

import static org.hamcrest.CoreMatchers.instanceOf;

import arrayUtils.ArrayUtilities;

public class FaultFactory {
	private int[] faultLabel = null;

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
		System.out.println(instanceOf(fault.getClass()));
		System.out.println(faultArray.length + "  " + hvPinDeFault.length + "  " + hvChannelDeFault.length + "  "
				+ hvConnectorDeFault.length + "  " + this.faultLabel.length);
		if (fault instanceof HVPinFault) {
			hvPinDeFault = faultArray;
			for (int i = 0; i < hvPinDeFault.length; i++) {
				System.out.print(hvPinDeFault[i] + "  ");
			}
			System.out.println(" \n \n ++++++ \n");
			System.arraycopy(hvPinDeFault, 0, this.faultLabel, 0, hvPinDeFault.length);
			for (int i = 0; i < this.faultLabel.length; i++) {
				System.out.print(this.faultLabel[i] + "  ");
			}
			System.out.println(" \n \n f1++++++ \n");
			System.arraycopy(hvChannelDeFault, 0, this.faultLabel, hvPinDeFault.length, hvChannelDeFault.length);
			for (int i = 0; i < this.faultLabel.length; i++) {
				System.out.print(this.faultLabel[i] + "  ");
			}
			System.out.println(" \n \n f2++++++ \n");
			System.arraycopy(hvConnectorDeFault, 0, this.faultLabel, hvChannelDeFault.length,
					hvConnectorDeFault.length);
			for (int i = 0; i < this.faultLabel.length; i++) {
				System.out.print(this.faultLabel[i] + "  ");
			}
			System.out.println(" \n \n f3++++++ \n");
			//
			// System.arraycopy(hvFuseDeFault, 0, this.faultLabel,
			// hvConnectorDeFault.length, hvFuseDeFault.length);
			//
			// System.arraycopy(hvDeadDeFault, 0, this.faultLabel,
			// hvFuseDeFault.length, hvDeadDeFault.length);
			//
			// System.arraycopy(hvHotDeFault, 0, this.faultLabel,
			// hvDeadDeFault.length, hvHotDeFault.length);

			for (int i = 0; i < this.faultLabel.length; i++) {
				System.out.print(this.faultLabel[i] + "  ");
			}
			System.out.println(" \n \n");
		} else if (fault instanceof HVChannelFault) {
			hvChannelDeFault = faultArray;
			System.arraycopy(hvPinDeFault, 0, this.faultLabel, 0, hvPinDeFault.length);
			System.arraycopy(hvChannelDeFault, 0, this.faultLabel, hvPinDeFault.length, hvChannelDeFault.length);
			System.arraycopy(hvConnectorDeFault, 0, this.faultLabel, hvChannelDeFault.length,
					hvConnectorDeFault.length);
			System.arraycopy(hvFuseDeFault, 0, this.faultLabel, hvConnectorDeFault.length, hvFuseDeFault.length);
			System.arraycopy(hvDeadDeFault, 0, this.faultLabel, hvFuseDeFault.length, hvDeadDeFault.length);
			System.arraycopy(hvHotDeFault, 0, this.faultLabel, hvDeadDeFault.length, hvHotDeFault.length);

		} else if (fault instanceof HVConnectorFault) {
			hvConnectorDeFault = faultArray;
			System.arraycopy(hvPinDeFault, 0, this.faultLabel, 0, hvPinDeFault.length);
			System.arraycopy(hvChannelDeFault, 0, this.faultLabel, hvPinDeFault.length, hvChannelDeFault.length);
			System.arraycopy(hvConnectorDeFault, 0, this.faultLabel, hvChannelDeFault.length,
					hvConnectorDeFault.length);
			System.arraycopy(hvFuseDeFault, 0, this.faultLabel, hvConnectorDeFault.length, hvFuseDeFault.length);
			System.arraycopy(hvDeadDeFault, 0, this.faultLabel, hvFuseDeFault.length, hvDeadDeFault.length);
			System.arraycopy(hvHotDeFault, 0, this.faultLabel, hvDeadDeFault.length, hvHotDeFault.length);
		} else if (fault instanceof HVFuseFault) {
			hvFuseDeFault = faultArray;
			System.arraycopy(hvPinDeFault, 0, this.faultLabel, 0, hvPinDeFault.length);
			System.arraycopy(hvChannelDeFault, 0, this.faultLabel, hvPinDeFault.length, hvChannelDeFault.length);
			System.arraycopy(hvConnectorDeFault, 0, this.faultLabel, hvChannelDeFault.length,
					hvConnectorDeFault.length);
			System.arraycopy(hvFuseDeFault, 0, this.faultLabel, hvConnectorDeFault.length, hvFuseDeFault.length);
			System.arraycopy(hvDeadDeFault, 0, this.faultLabel, hvFuseDeFault.length, hvDeadDeFault.length);
			System.arraycopy(hvHotDeFault, 0, this.faultLabel, hvDeadDeFault.length, hvHotDeFault.length);
		} else if (fault instanceof HVDeadWire) {
			hvDeadDeFault = faultArray;
			System.arraycopy(hvPinDeFault, 0, this.faultLabel, 0, hvPinDeFault.length);
			System.arraycopy(hvChannelDeFault, 0, this.faultLabel, hvPinDeFault.length, hvChannelDeFault.length);
			System.arraycopy(hvConnectorDeFault, 0, this.faultLabel, hvChannelDeFault.length,
					hvConnectorDeFault.length);
			System.arraycopy(hvFuseDeFault, 0, this.faultLabel, hvConnectorDeFault.length, hvFuseDeFault.length);
			System.arraycopy(hvDeadDeFault, 0, this.faultLabel, hvFuseDeFault.length, hvDeadDeFault.length);
			System.arraycopy(hvHotDeFault, 0, this.faultLabel, hvDeadDeFault.length, hvHotDeFault.length);
		} else if (fault instanceof HVHotWire) {
			hvHotDeFault = faultArray;
			System.arraycopy(hvPinDeFault, 0, this.faultLabel, 0, hvPinDeFault.length);
			System.arraycopy(hvChannelDeFault, 0, this.faultLabel, hvPinDeFault.length, hvChannelDeFault.length);
			System.arraycopy(hvConnectorDeFault, 0, this.faultLabel, hvChannelDeFault.length,
					hvConnectorDeFault.length);
			System.arraycopy(hvFuseDeFault, 0, this.faultLabel, hvConnectorDeFault.length, hvFuseDeFault.length);
			System.arraycopy(hvDeadDeFault, 0, this.faultLabel, hvFuseDeFault.length, hvDeadDeFault.length);
			System.arraycopy(hvHotDeFault, 0, this.faultLabel, hvDeadDeFault.length, hvHotDeFault.length);
		}
	}

	private int[] makeFuseDefaultLabel(int length) {
		int[] deFault = new int[length];
		for (int i = 0; i < deFault.length; i++) {
			deFault[i] = 0;
		}
		return deFault;
	}
}// end of FaultFactory class.
