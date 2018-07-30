package faulttypes;

import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.ThreadLocalRandom;
import java.util.stream.IntStream;

import org.apache.commons.lang3.tuple.Pair;

import utils.ArrayUtilities;

public class HVConnectorFault extends FaultData {

	private Map<Integer, Pair<Integer, Integer>> eMap = new HashMap<>();
	private Map<Integer, Pair<Integer, Integer>> treeMap = new HashMap<>();
	private Map<Integer, Pair<Integer, Integer>> threeMap = new HashMap<>();
	private Map<Integer, Pair<Integer, Integer>> rndPair;

	public HVConnectorFault() {
		setupBundles();
		this.xRnd = ThreadLocalRandom.current().nextInt(1, 113);
		this.yRnd = ThreadLocalRandom.current().nextInt(1, 7);
		this.faultLocation = findFaultLocation();

		this.label = ArrayUtilities.hvConnectorFault;
		this.rndPair = getRandomPair();
		/**
		 * reducedLabel is initialized as a IntStream in findFaultLocation
		 */
		makeDataSet();
		makeFaultArray();

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
			return modifyMap(xPlace, eMap);
		case 2:
			return modifyMap(xPlace, eMap);
		case 3:
			if (yBin == 3 || yBin == 5) {
				return modifyMap(xPlace, treeMap);
			} else {
				return modifyMap(xPlace, eMap);
			}
		case 4:
			return modifyMap(xPlace, treeMap);
		case 5:
			return modifyMap(xPlace, treeMap);
		case 6:
			if (yBin == 2 || yBin == 4) {
				return modifyMap(xPlace, treeMap);
			} else {
				return modifyMap(xPlace, threeMap);
			}
		case 7:
			return modifyMap(xPlace, threeMap);
		case 8:
			return modifyMap(xPlace, threeMap);
		default:
			return null;
		}
	}

	@Override
	protected void makeDataSet() {
		// Map<Integer, Pair<Integer, Integer>> rndPair = getRandomPair();
		for (int i = 0; i < data[0].length; i++) { // i are the rows (layers)
			for (int j = 0; j < data.length; j++) { // j are the columns (wires)
				if (j <= this.rndPair.get(i + 1).getRight() - 1 && j >= this.rndPair.get(i + 1).getLeft() - 1) {
					data[j][i] = makeRandomData(faultRangeMin, faultRangeMax);
				} else {
					data[j][i] = makeRandomData(rangeMin, rangeMax);
				}
			}
		}
	}

	private void makeFaultArray() {
		for (int i = 0; i < this.label.length; i++) {
			if (i == faultLocation) {
				this.label[i] = 1;
			} else {
				this.label[i] = 0;
			}
		}
	}

	private int makeRandomData(int rangeMin, int rangeMax) {
		return ThreadLocalRandom.current().nextInt(rangeMin, rangeMax + 1);
	}

	private Map<Integer, Pair<Integer, Integer>> getRandomPair() {
		return this.findWireRange(this.xRnd, this.yRnd);
	}

	private void setupBundles() {
		setupEMap();
		setupTreeMap();
		setupThreeMap();

	}

	private int findFaultLocation() {
		int placer;
		int xPlace;
		int iInc;

		xPlace = (this.xRnd - 1) / 8 * 8;
		iInc = xPlace / 8;
		placer = (this.xRnd + iInc) % 9;
		int test = xPlace / 8;

		switch (placer) {
		case 1:
		case 2:
			this.reducedLabel = IntStream.of(1, 0, 0).toArray();
			this.faultName = FaultNames.CONNECTOR_E;
			return 0 + 3 * test;
		case 3:
			if (this.yRnd == 3 || this.yRnd == 5) {
				this.reducedLabel = IntStream.of(0, 1, 0).toArray();
				this.faultName = FaultNames.CONNECTOR_TREE;
				return 1 + 3 * test;
			} else {
				this.reducedLabel = IntStream.of(1, 0, 0).toArray();
				this.faultName = FaultNames.CONNECTOR_E;
				return 0 + 3 * test;
			}
		case 4:
		case 5:
			this.reducedLabel = IntStream.of(0, 1, 0).toArray();
			this.faultName = FaultNames.CONNECTOR_TREE;
			return 1 + 3 * test;
		case 6:
			if (this.yRnd == 2 || this.yRnd == 4) {
				this.reducedLabel = IntStream.of(0, 1, 0).toArray();
				this.faultName = FaultNames.CONNECTOR_TREE;
				return 1 + 3 * test;
			} else {
				this.reducedLabel = IntStream.of(0, 0, 1).toArray();
				this.faultName = FaultNames.CONNECTOR_THREE;
				return 2 + 3 * test;

			}
		case 7:
		case 8:
			this.reducedLabel = IntStream.of(0, 0, 1).toArray();
			this.faultName = FaultNames.CONNECTOR_THREE;
			return 2 + 3 * test;
		default:
			return -1;
		}
	}

	public static void main(String[] args) {
		for (int i = 0; i < 5; i++) {
			FaultData faultData = new HVConnectorFault();
			faultData.plotData();
			System.out.println(faultData.getFaultLocation() + "  " + Arrays.toString(faultData.getReducedLabel()));

		}

	}

}
