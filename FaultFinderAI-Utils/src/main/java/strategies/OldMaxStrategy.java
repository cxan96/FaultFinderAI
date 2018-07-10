package strategies;

import org.nd4j.linalg.api.ndarray.INDArray;

import utils.FaultUtils;

public class OldMaxStrategy implements FaultRecordScalerStrategy {

	@Override
	public void normalize(INDArray features) {
		double maxRange = FaultUtils.RANGE_MAX;
		features.divi((maxRange));
	}
}
