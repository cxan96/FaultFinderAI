package clasDC.faults;

import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.ThreadLocalRandom;

import org.apache.commons.lang3.tuple.Pair;

public class HVFuseFault extends FaultData {

	private Map<Integer, Pair<Integer, Integer>> eMap = new HashMap<>();
	private Map<Integer, Pair<Integer, Integer>> treeMap = new HashMap<>();
	private Map<Integer, Pair<Integer, Integer>> threeMap = new HashMap<>();

	private Map<Integer, Pair<Integer, Integer>> bundleA = new HashMap<>();
	private Map<Integer, Pair<Integer, Integer>> bundleB = new HashMap<>();
	private Map<Integer, Pair<Integer, Integer>> bundleC = new HashMap<>();

	public HVFuseFault() {
		setupBundles();
		this.xRnd = ThreadLocalRandom.current().nextInt(1, 113);
		this.yRnd = ThreadLocalRandom.current().nextInt(1, 7);
		this.faultyWires = getRandomPair();
	}

	public HVFuseFault(FaultNames faultName) {
		setupBundles();
		this.xRnd = ThreadLocalRandom.current().nextInt(1, 113);
		this.yRnd = ThreadLocalRandom.current().nextInt(1, 7);
		this.faultyWires = getRandomPair();

		while (this.faultName != faultName) {
			this.xRnd = ThreadLocalRandom.current().nextInt(1, 113);
			this.yRnd = ThreadLocalRandom.current().nextInt(1, 7);
			this.faultyWires = getRandomPair();
		}

	}

	/**
	 * @param xplace
	 */
	public HVFuseFault(FaultNames faultName, int xplace) {
		setupBundles();
		this.faultName = faultName;
		this.xRnd = xplace;
		this.yRnd = ThreadLocalRandom.current().nextInt(1, 7);
		getCorrectBundle();
	}

	/**
	 * getCorrectBundle will set the this.faultyWires correctly depending on the
	 * FaultNames and the xplace the xplace is the location of the bundle 1 - 14
	 * location for each bundle
	 */
	private void getCorrectBundle() {

		if (faultName.equals(FaultNames.FUSE_A)) {
			this.faultyWires = modifyMap(this.xRnd * 16, this.bundleA);
		} else if (faultName.equals(FaultNames.FUSE_B)) {
			this.faultyWires = modifyMap(this.xRnd * 16, this.bundleB);
		} else if (faultName.equals(FaultNames.FUSE_C)) {
			this.faultyWires = modifyMap(this.xRnd * 16, this.bundleC);
		} else {
			throw new IllegalAccessError("This " + faultName + " is not defined");
		}
	}

	private Map<Integer, Pair<Integer, Integer>> getRandomPair() {
		return this.findWireRange(this.xRnd, this.yRnd);
	}

	private static Map<Integer, Pair<Integer, Integer>> modifyMap(int xPlace,
			Map<Integer, Pair<Integer, Integer>> aMap) {

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

		xPlace = (xBin - 1) / 16 * 16;
		iInc = xPlace / 8;
		placer = (xBin + iInc) % 18;
		switch (placer) {
		case 1:
		case 2:
		case 3:
		case 4:
		case 5:
			this.faultName = FaultNames.FUSE_A;
			return modifyMap(xPlace, bundleA);
		case 6:
			if (yBin == 2 || yBin == 4) {
				this.faultName = FaultNames.FUSE_A;
				return modifyMap(xPlace, bundleA);
			} else {
				this.faultName = FaultNames.FUSE_B;
				return modifyMap(xPlace, bundleB);
			}
		case 7:
		case 8:
		case 9:
		case 10:
			this.faultName = FaultNames.FUSE_B;
			return modifyMap(xPlace, bundleB);
		case 11:
			if (yBin == 3 || yBin == 5) {
				this.faultName = FaultNames.FUSE_C;
				return modifyMap(xPlace, bundleC);
			} else {
				this.faultName = FaultNames.FUSE_B;
				return modifyMap(xPlace, bundleB);
			}
		case 12:
		case 13:
		case 14:
		case 15:
		case 16:
			this.faultName = FaultNames.FUSE_C;
			return modifyMap(xPlace, bundleC);
		default:
			return null;
		}
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

	private void setupBundleA() {
		for (int i = 1; i <= 6; i++) {
			bundleA.put(i, Pair.of(this.eMap.get(i).getLeft(), this.treeMap.get(i).getRight()));
		}
	}

	private void setupBundleB() {
		for (int i = 1; i <= 6; i++) {
			bundleB.put(i, Pair.of(this.threeMap.get(i).getLeft(), 8 + this.eMap.get(i).getRight()));
		}
	}

	private void setupBundleC() {
		for (int i = 1; i <= 6; i++) {
			bundleC.put(i, Pair.of(8 + this.treeMap.get(i).getLeft(), 8 + this.threeMap.get(i).getRight()));
		}
	}

	public void setupBundles() {
		setupEMap();
		setupTreeMap();
		setupThreeMap();
		setupBundleA();
		setupBundleB();
		setupBundleC();

	}

	@Override
	public Fault getInformation() {
		return new Fault(this.getClass().getSimpleName(), this.faultName, this.faultyWires,
				Pair.of(this.xRnd, this.yRnd));
	}

}
