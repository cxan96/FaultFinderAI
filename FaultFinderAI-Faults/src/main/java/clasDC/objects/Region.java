/**
 * 
 */
package clasDC.objects;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.Stream;

import clasDC.faults.FaultNames;
import lombok.Builder;
import lombok.Getter;

/**
 * @author m.c.kunkel
 *
 */
@Getter
public class Region extends CLASObject {
	private int region;

	@Builder
	private Region(int region, int nchannels, int minFaults, int maxFaults, List<FaultNames> desiredFaults,
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

		this.objectType = "Region";
		this.height = 72;
		setPriors();

	}

	public static void main(String[] args) {
		CLASObject object = Region.builder().region(3).nchannels(1).maxFaults(10)
				.desiredFaults(Stream.of(FaultNames.CONNECTOR_TREE, FaultNames.CONNECTOR_TREE, FaultNames.CHANNEL_THREE,
						FaultNames.PIN_SMALL).collect(Collectors.toCollection(ArrayList::new)))
				.singleFaultGen(false).build();
		System.out.println(object.getHeight() + "  " + object.getWidth() + "  " + object.getNchannels());
		List<FaultNames> listDistinct = object.getDesiredFaults().stream().distinct().collect(Collectors.toList());
		System.out.println(listDistinct.size() + "  " + object.getDesiredFaults().size() + "   "
				+ object.getDesiredFaults().stream().distinct().collect(Collectors.toList()).size());

		if (object instanceof Region) {
			Region region = (Region) object;
			System.out
					.println(region.getRegion() + "   looking glass  " + region.getHeight() + "  " + region.getWidth());

		}
		System.out.println(Arrays.deepToString(object.getPriors()));

	}

}
