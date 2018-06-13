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
		}
		return null;
	}

}// end of FaultFactory class.
