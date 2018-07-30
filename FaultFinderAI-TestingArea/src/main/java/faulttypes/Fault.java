package faulttypes;

import java.util.Map;

import org.apache.commons.lang3.tuple.Pair;

public class Fault {
	/**
	 * Map<layerLocation, Pair<leftWire,rightWire>>
	 */
	private Map<Integer, Pair<Integer, Integer>> wireInfo;
	private String faultName;
	private FaultNames subFaultName;

	public Fault(String faultName, FaultNames subFaultName, Map<Integer, Pair<Integer, Integer>> wireInfo) {
		this.faultName = faultName;
		this.subFaultName = subFaultName;
		this.wireInfo = wireInfo;
	}

	public Map<Integer, Pair<Integer, Integer>> getWireInfo() {
		return wireInfo;
	}

	public String getFaultName() {
		return faultName;
	}

	public FaultNames getSubFaultName() {
		return subFaultName;
	}

	public boolean compareFault(Fault anotherFault) {

		for (Map.Entry<Integer, Pair<Integer, Integer>> entry : this.getWireInfo().entrySet()) {
			Integer key = entry.getKey();
			Pair<Integer, Integer> value = entry.getValue();
			for (Map.Entry<Integer, Pair<Integer, Integer>> anotherEntry : anotherFault.getWireInfo().entrySet()) {
				Integer anotherKey = anotherEntry.getKey();
				Pair<Integer, Integer> anotherValue = anotherEntry.getValue();
				if (anotherKey.equals(key)) {
					// System.out.println("Investigating anotherFault Layer " +
					// anotherKey + " with left: "
					// + anotherValue.getLeft() + " with right: " +
					// anotherValue.getRight());
					// System.out.println("thisFault Layer " + key + " with
					// left: " + value.getLeft() + " with right: "
					// + value.getRight());

					// now check to see if the new fault is inside the existing
					// fault
					if (anotherValue.getLeft() >= value.getLeft() && anotherValue.getLeft() <= value.getRight()) {
						return false;
					} // this works
						// lets make 2 statements so its more readable
					if (anotherValue.getRight() >= value.getLeft() && anotherValue.getRight() <= value.getRight()) {
						return false;
					}
					if (anotherValue.getRight() >= value.getRight() && anotherValue.getLeft() <= value.getLeft()) {
						return false;
					}
				}
			}
		}
		return true;
	}

}
