package faulttypes;

public class FaultFactory {

	public FaultFactory() {

	}

	// use getPlan method to get object of type Plan
	public FaultData getFault(int type) {
		if (type == 0) {
			return new HVPinFault();
		} else if (type == 1) {
			return new HVChannelFault();
		} else if (type == 2) {
			return new HVConnectorFault();
		} else if (type == 3) {
			return new HVFuseFault();
		} else if (type == 4) {
			return new HVDeadWire();
		} else if (type == 5) {
			return new HVHotWire();
		}
		return null;
	}

}// end of FaultFactory class.
