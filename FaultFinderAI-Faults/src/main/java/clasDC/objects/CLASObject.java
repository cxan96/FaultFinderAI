/**
 * 
 */
package clasDC.objects;

import java.util.List;
import java.util.stream.Collectors;

import clasDC.faults.FaultNames;
import lombok.Getter;
import utils.FaultUtils;

/**
 * @author m.c.kunkel
 *
 */
@Getter
public abstract class CLASObject {

	protected String objectType;
	protected int height;
	protected final int width = 112;

	protected int nchannels;
	protected int maxFaults;
	protected List<FaultNames> desiredFaults = null;
	protected boolean singleFaultGen;

	protected double[][] priors;

	protected void setPriors() {
		List<FaultNames> listDistinct = this.desiredFaults.stream().distinct().collect(Collectors.toList());
		priors = listDistinct.get(0).getPrior();
		for (int i = 1; i < listDistinct.size(); i++) {
			priors = FaultUtils.merge(priors, listDistinct.get(i).getPrior());
		}

	}

}
