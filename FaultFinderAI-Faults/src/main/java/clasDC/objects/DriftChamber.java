/**
 * 
 */
package clasDC.objects;

import java.util.List;

import clasDC.faults.FaultNames;
import lombok.Builder;
import lombok.Getter;

@Getter
public class DriftChamber extends CLASObject {
	private int region;

	@Builder
	private DriftChamber(int region, int nchannels, int minFaults, int maxFaults, List<FaultNames> desiredFaults,
			boolean singleFaultGen, ContainerType containerType) {
		if (region > 3 || region < 1) {
			throw new IllegalArgumentException("Invalid input: (region), must have values less than"
					+ " (4) and more than (0). Received: (" + region + ")");
		}
		this.region = region;
		this.nchannels = nchannels;
		this.minFaults = minFaults;
		this.maxFaults = maxFaults;
		this.desiredFaults = desiredFaults;
		this.singleFaultGen = singleFaultGen;
		this.containerType = containerType;

		this.objectType = "DriftChamber";
		this.height = 12;
		setPriors();

	}
}
