package client;

import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.ThreadLocalRandom;

import org.nd4j.linalg.primitives.Pair;

import faulttypes.FaultFactory;

public class ViewRandomFaults {
	public static void main(String[] args) {

		int numPlots = 10;
		FaultFactory factory = new FaultFactory();
		Map<Integer, Integer> aMap = new HashMap<>();
		for (int i = 0; i < numPlots; i++) {
			int rnd = ThreadLocalRandom.current().nextInt(6);
			int rnd2 = checkMap(aMap, rnd).getLeft();
			factory.getFault(rnd2).plotData();

		}

	}

	public static Pair<Integer, Map<Integer, Integer>> checkMap(Map<Integer, Integer> aMap, int rnd) {
		if (aMap.containsKey(rnd)) {
			if (aMap.get(rnd) == 1) {
				rnd = ThreadLocalRandom.current().nextInt(6);
				checkMap(aMap, rnd);
			} else {
				aMap.put(rnd, 1);
				return Pair.of(rnd, aMap);
			}
		} else {
			aMap.put(rnd, 1);
		}
		return Pair.of(rnd, aMap);
	}
}
