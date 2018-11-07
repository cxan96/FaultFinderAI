package clasDC;

/**
 * 
 * @author m.c.kunkel <br>
 *         a CLASDriftChamber contains 2 superlayers 1-2, 3-4, 5-6. Each
 *         superlayer contains 6 layers and 112 wires. <br>
 *         FaultFactory provides the layers and wire faults.
 * 
 */

public class CLASDriftChamber {
	private int region;
	private int sector;

	public CLASDriftChamber(int region, int sector) {
		this.region = region;
		this.sector = sector;
	}

	public int getRegion() {
		return region;
	}

	public void setRegion(int region) {
		this.region = region;
	}

	public int getSector() {
		return sector;
	}

	public void setSector(int sector) {
		this.sector = sector;
	}

}
