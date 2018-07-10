package strategies;

import org.nd4j.linalg.api.ndarray.INDArray;

public class MinMaxStrategy implements FaultRecordScalerStrategy {

	private double percent;

	public MinMaxStrategy() {
		this.percent = 0.0;
	}

	public MinMaxStrategy(double percent) {
		this.percent = percent;
	}

	@Override
	public void normalize(INDArray features) {

		double maxRange = (double) features.maxNumber();

		double minRange = (double) features.minNumber();
		double minMaxBias = percent * (maxRange - minRange);

		if (minRange != 0)
			features.subi(minRange); // Offset by minRange
		features.divi((2.0 * minMaxBias + maxRange - minRange));

	}

}
