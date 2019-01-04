package clasDC.faults;

import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.ThreadLocalRandom;

import org.apache.commons.lang3.tuple.Pair;

public class HVConnectorFault extends FaultData {

	private Map<Integer, Pair<Integer, Integer>> eMap = new HashMap<>();
	private Map<Integer, Pair<Integer, Integer>> treeMap = new HashMap<>();
	private Map<Integer, Pair<Integer, Integer>> threeMap = new HashMap<>();

	public HVConnectorFault() {
		setupBundles();
		this.xRnd = ThreadLocalRandom.current().nextInt(1, 113);
		this.yRnd = ThreadLocalRandom.current().nextInt(1, 7);
		this.faultyWires = getRandomPair();
	}

	public HVConnectorFault(FaultNames faultName) {
		setupBundles();
		while (this.faultName != faultName) {
			this.xRnd = ThreadLocalRandom.current().nextInt(1, 113);
			this.yRnd = ThreadLocalRandom.current().nextInt(1, 7);
			this.faultyWires = getRandomPair();
		}

	}

	public HVConnectorFault(FaultNames faultName, int xplace) {
		setupBundles();
		// For completeness
		this.yRnd = 1;
		this.xRnd = xplace;

		this.faultName = faultName;
		getCorrectBundle();
	}

	/**
	 * getCorrectBundle will set the this.faultyWires correctly depending on the
	 * FaultNames and the xplace the xplace is the location of the bundle 1 - 14
	 * location for each bundle
	 */
	private void getCorrectBundle() {

		if (faultName.equals(FaultNames.CONNECTOR_E)) {
			this.faultyWires = modifyMap(this.xRnd * 8, this.eMap);
		} else if (faultName.equals(FaultNames.CONNECTOR_TREE)) {
			this.faultyWires = modifyMap(this.xRnd * 8, this.treeMap);
		} else if (faultName.equals(FaultNames.CONNECTOR_THREE)) {
			this.faultyWires = modifyMap(this.xRnd * 8, this.threeMap);
		} else {
			throw new IllegalAccessError("This " + faultName + " is not defined");
		}
	}

	private void setupBundles() {
		setupEMap();
		setupTreeMap();
		setupThreeMap();
	}

	private void setupEMap() {
		eMap.put(1, Pair.of(1, 3));
		eMap.put(2, Pair.of(1, 3));
		eMap.put(3, Pair.of(1, 2));
		eMap.put(4, Pair.of(1, 3));
		eMap.put(5, Pair.of(1, 2));
		eMap.put(6, Pair.of(1, 3));

	}

	private void setupTreeMap() {
		treeMap.put(1, Pair.of(4, 5));
		treeMap.put(2, Pair.of(4, 6));
		treeMap.put(3, Pair.of(3, 5));
		treeMap.put(4, Pair.of(4, 6));
		treeMap.put(5, Pair.of(3, 5));
		treeMap.put(6, Pair.of(4, 5));
	}

	private void setupThreeMap() {
		threeMap.put(1, Pair.of(6, 8));
		threeMap.put(2, Pair.of(7, 8));
		threeMap.put(3, Pair.of(6, 8));
		threeMap.put(4, Pair.of(7, 8));
		threeMap.put(5, Pair.of(6, 8));
		threeMap.put(6, Pair.of(6, 8));
	}

	private Map<Integer, Pair<Integer, Integer>> modifyMap(int xPlace, Map<Integer, Pair<Integer, Integer>> aMap) {

		Map<Integer, Pair<Integer, Integer>> aNewMap = new HashMap<>();

		for (Map.Entry<Integer, Pair<Integer, Integer>> entry : aMap.entrySet()) {
			Integer key = entry.getKey();
			Pair<Integer, Integer> value = entry.getValue();
			aNewMap.put(key, Pair.of(value.getLeft() + xPlace, value.getRight() + xPlace));
		}
		return aNewMap;
	}

	public Map<Integer, Pair<Integer, Integer>> findWireRange(int xBin, int yBin) {
		int placer;
		int xPlace;
		int iInc;

		xPlace = (xBin - 1) / 8 * 8;
		iInc = xPlace / 8;
		placer = (xBin + iInc) % 9;
		switch (placer) {
		case 1:
		case 2:
			this.faultName = FaultNames.CONNECTOR_E;
			return modifyMap(xPlace, eMap);
		case 3:
			if (yBin == 3 || yBin == 5) {
				this.faultName = FaultNames.CONNECTOR_TREE;
				return modifyMap(xPlace, treeMap);
			} else {
				this.faultName = FaultNames.CONNECTOR_E;
				return modifyMap(xPlace, eMap);
			}
		case 4:
		case 5:
			this.faultName = FaultNames.CONNECTOR_TREE;
			return modifyMap(xPlace, treeMap);
		case 6:
			if (yBin == 2 || yBin == 4) {
				this.faultName = FaultNames.CONNECTOR_TREE;
				return modifyMap(xPlace, treeMap);
			} else {
				this.faultName = FaultNames.CONNECTOR_THREE;
				return modifyMap(xPlace, threeMap);
			}
		case 7:
		case 8:
			this.faultName = FaultNames.CONNECTOR_THREE;
			return modifyMap(xPlace, threeMap);
		default:
			return null;
		}
	}

	private Map<Integer, Pair<Integer, Integer>> getRandomPair() {
		return this.findWireRange(this.xRnd, this.yRnd);
	}

	@Override
	public Fault getInformation() {
		return new Fault(this.getClass().getSimpleName(), this.faultName, this.faultyWires,
				Pair.of(this.xRnd, this.yRnd));
	}

}
