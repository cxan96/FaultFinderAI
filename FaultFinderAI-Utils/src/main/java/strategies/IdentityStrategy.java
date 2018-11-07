/**
 * 
 */
package strategies;

import org.nd4j.linalg.api.ndarray.INDArray;

/**
 * @author m.c.kunkel
 *
 */
public class IdentityStrategy implements FaultRecordScalerStrategy {

	public IdentityStrategy() {
		// TODO Auto-generated constructor stub
	}

	@Override
	public void normalize(INDArray features) {
		features.muli(1);
	}

}
