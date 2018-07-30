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

}
