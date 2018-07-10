package strategies;

import org.nd4j.linalg.api.ndarray.INDArray;

public class StandardizeMinMax implements FaultRecordScalerStrategy {

	double percent;

	public StandardizeMinMax() {
		this.percent = 0.0;
	}

	public StandardizeMinMax(double percent) {
		this.percent = percent;
	}

	@Override
	public void normalize(INDArray features) {
		new StandardScore().normalize(features);
		new MinMaxStrategy(percent).normalize(features);
	}

}
