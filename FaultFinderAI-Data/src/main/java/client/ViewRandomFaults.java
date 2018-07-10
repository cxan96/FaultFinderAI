package client;

import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.ThreadLocalRandom;

import faulttypes.FaultFactory;
import strategies.SigmoidStrategy;

public class ViewRandomFaults {
	public static void main(String[] args) {

		int numPlots = 10;
		FaultFactory factory = new FaultFactory();
		Map<Integer, Integer> aMap = new HashMap<>();
		for (int i = 0; i < 1; i++) {
			int rnd = ThreadLocalRandom.current().nextInt(6);
			factory.getFault(rnd).plotData(new SigmoidStrategy());// new
																	// SigmoidStrategy()
		}
	}
}
