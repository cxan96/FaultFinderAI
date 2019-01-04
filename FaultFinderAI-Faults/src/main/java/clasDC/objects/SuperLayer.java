/**
 * 
 */
package clasDC.objects;

import java.util.List;

import clasDC.faults.FaultNames;
import lombok.Builder;
import lombok.Getter;

/**
 * @author m.c.kunkel
 *
 */
@Getter
public class SuperLayer extends CLASObject {

	private int superlayer;

	@Builder
	private SuperLayer(int superlayer, int nchannels, int minFaults, int maxFaults, FaultNames desiredFault,
			double desiredFaultGenRate, List<FaultNames> desiredFaults, boolean singleFaultGen, boolean isScaled,
			ContainerType containerType) {
		if (superlayer > 6 || superlayer < 1) {
			throw new IllegalArgumentException("Invalid input: (superlayer), must have values less than"
					+ " ( 7) and more than (0). Received: (" + superlayer + ")");
		}
		this.superlayer = superlayer;
		this.nchannels = nchannels;
		this.minFaults = minFaults;
		this.maxFaults = maxFaults;
		this.desiredFault = desiredFault;
		this.desiredFaultGenRate = desiredFaultGenRate;
		this.desiredFaults = desiredFaults;
		this.singleFaultGen = singleFaultGen;
		this.isScaled = isScaled;
		this.containerType = containerType;

		this.objectType = "SuperLayer";
		this.height = 6;
		setPriors();

	}
}
