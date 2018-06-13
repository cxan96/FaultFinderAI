package faulttypes;

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
		makeDataSet();
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
		for (int i = 0; i < data[0].length; i++) { // i are the rows (layers)
			for (int j = 0; j < data.length; j++) { // j are the columns (wires)
				if (j <= getRandomPair().get(i + 1).getRight() - 1 && j >= getRandomPair().get(i + 1).getLeft() - 1) {
					data[j][i] = makeRandomData(faultRangeMin, faultRangeMax);// -
																				// (int)
																				// (0.1
																				// *
																				// faultRangeMax)
				} else {
					data[j][i] = makeRandomData(rangeMin, rangeMax);
				}
				// System.out.print(data[j][i] + "\t");
				// if (j == data.length - 1) {
				// System.out.println("");
				// }
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

	public static void main(String[] args) {
		HVConnectorFault hvConnectorFault = new HVConnectorFault();
		hvConnectorFault.plotData();
	}
}
