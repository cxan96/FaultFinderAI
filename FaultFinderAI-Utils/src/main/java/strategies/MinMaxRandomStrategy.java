package strategies;

import java.util.concurrent.ThreadLocalRandom;

import org.nd4j.linalg.api.ndarray.INDArray;

public class MinMaxRandomStrategy implements FaultRecordScalerStrategy {

	private double randomRange;

	public MinMaxRandomStrategy() {
		this.randomRange = 0.0;
	}

	public MinMaxRandomStrategy(double randomRange) {
		this.randomRange = randomRange;
	}

	@Override
	public void normalize(INDArray features) {

		double range = ThreadLocalRandom.current().nextDouble(0.0, randomRange);
		double maxRange = (double) features.maxNumber();

		double minRange = (double) features.minNumber();
		double minMaxBias = range * (maxRange - minRange);

		if (minRange != 0)
			features.subi(minRange); // Offset by minRange
		features.divi((2.0 * minMaxBias + maxRange - minRange));

	}

}
