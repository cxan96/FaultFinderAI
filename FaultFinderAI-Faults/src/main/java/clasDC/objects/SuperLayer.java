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
	private SuperLayer(int superlayer, int nchannels, int maxFaults, List<FaultNames> desiredFaults,
			boolean singleFaultGen) {
		if (superlayer > 6 || superlayer < 1) {
			throw new IllegalArgumentException("Invalid input: (superlayer), must have values less than"
					+ " ( 7) and more than (0). Received: (" + superlayer + ")");
		}
		this.superlayer = superlayer;
		this.nchannels = nchannels;
		this.maxFaults = maxFaults;
		this.desiredFaults = desiredFaults;
		this.singleFaultGen = singleFaultGen;

		this.objectType = "SuperLayer";
		this.height = 6;
		setPriors();

	}
}
