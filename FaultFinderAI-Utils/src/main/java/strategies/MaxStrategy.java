package strategies;

import org.nd4j.linalg.api.ndarray.INDArray;

public class MaxStrategy implements FaultRecordScalerStrategy {
	private double percent;

	public MaxStrategy() {
		this.percent = 0.0;
	}

	public MaxStrategy(double percent) {
		this.percent = percent;
	}

	@Override
	public void normalize(INDArray features) {
		double maxRange = (double) features.maxNumber();
		double minMaxBias = percent * maxRange;

		features.divi((maxRange + minMaxBias));
	}
}
