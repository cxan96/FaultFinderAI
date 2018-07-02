package client;

import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.ThreadLocalRandom;

import faulttypes.FaultFactory;

public class ViewRandomFaults {
	public static void main(String[] args) {

		int numPlots = 10;
		FaultFactory factory = new FaultFactory();
		Map<Integer, Integer> aMap = new HashMap<>();
		for (int i = 0; i < numPlots; i++) {
			int rnd = ThreadLocalRandom.current().nextInt(6);
			factory.getFault(rnd).plotData();
		}
	}
}
