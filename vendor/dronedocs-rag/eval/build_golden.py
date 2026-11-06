"""Source of truth for the golden evaluation set -> eval/golden.jsonl.

Every answerable item carries a verbatim ``evidence`` quote copied from the source document.
``scripts/verify_golden.py`` re-loads the raw documents and fails if any quote is not found on
the cited page / section, so the reference answers stay verified as the corpus evolves.
Unanswerable items (answerable=False) test refusal: the corpus does not contain the answer.
"""

import json
from pathlib import Path

R21 = "drone_rules_2021.pdf"
A22 = "drone_amendment_rules_2022.pdf"
A23 = "drone_amendment_rules_2023.pdf"
PIB = "pib_drone_rules_backgrounder.pdf"
AP_ARM = "ardupilot_copter_arming_the_motors.html"
AP_BATT = "ardupilot_copter_failsafe-battery.html"
AP_RADIO = "ardupilot_copter_radio-failsafe.html"
AP_RTL = "ardupilot_copter_rtl-mode.html"
AP_LAND = "ardupilot_copter_land-mode.html"
AP_MODES = "ardupilot_copter_flight-modes.html"
PX_SAFE = "px4_config_safety.md"
PX_ARM = "px4_advanced_config_prearm_arm_disarm.md"
PX_POS = "px4_flight_modes_mc_position.md"
PX_LAND = "px4_flight_modes_mc_land.md"
PX_RTN = "px4_flight_modes_return.md"

# (id, question, reference answer, source, locator, verbatim evidence)
ANSWERABLE = [
    # ---------------- Drone Rules, 2021 (Gazette of India, G.S.R. 589(E)) ----------------
    (
        "dr-01",
        "What is the maximum all-up weight of a nano unmanned aircraft system under the Drone Rules 2021?",
        "A nano unmanned aircraft system weighs less than or equal to 250 grams.",
        R21,
        "page 27",
        "Nano unmanned aircraft system: weighing less than or equal to 250 grams",
    ),
    (
        "dr-02",
        "What weight range defines a micro drone under the Drone Rules 2021?",
        "A micro UAS weighs more than 250 grams but less than or equal to 2 kilograms.",
        R21,
        "page 27",
        "Micro unmanned aircraft system: weighing more than 250 grams, but less than or equal to 2",
    ),
    (
        "dr-03",
        "Which weight class is a 30 kg drone in under the Drone Rules 2021?",
        "Medium: more than 25 kg but less than or equal to 150 kg.",
        R21,
        "page 28",
        "Medium unmanned aircraft system: weighing more than 25 kilograms, but less than or equal to\n150 kilograms",
    ),
    (
        "dr-04",
        "Above what weight is an unmanned aircraft system classified as large?",
        "A large UAS weighs more than 150 kilograms.",
        R21,
        "page 28",
        "Large unmanned aircraft system: weighing more than 150 kilograms",
    ),
    (
        "dr-05",
        "Up to what height does the green zone extend under the Drone Rules 2021?",
        "Up to a vertical distance of 400 feet or 120 metres (and 200 feet / 60 m in the 8-12 km band around an operational airport).",
        R21,
        "page 26",
        "upto a vertical distance of 400 feet or 120 metre",
    ),
    (
        "dr-06",
        "Do I need prior permission to fly a drone in a green zone?",
        "No prior permission is required in a green zone, subject to the mandatory pre-flight verification of zonal restrictions in rule 21.",
        R21,
        "page 30",
        "No prior permission shall be required for operating an unmanned aircraft system in a green zone",
    ),
    (
        "dr-07",
        "What is the minimum and maximum age to be eligible for a remote pilot licence?",
        "The individual must be not less than 18 years and not more than 65 years of age.",
        R21,
        "page 31",
        "is not less than eighteen years of age and not more than sixty-five years of age",
    ),
    (
        "dr-08",
        "What educational qualification is required to get a remote pilot licence under the Drone Rules 2021?",
        "The individual must have passed the class tenth examination or its equivalent from a recognised Board.",
        R21,
        "page 31",
        "has passed class tenth examination or its equivalent from a recognised Board",
    ),
    (
        "dr-09",
        "How long is a remote pilot licence valid?",
        "Unless suspended or cancelled, it remains valid for ten years (renewable for up to ten years at a time).",
        R21,
        "page 32",
        "remain valid for a period of ten years",
    ),
    (
        "dr-10",
        "Which drones are exempt from needing a remote pilot licence?",
        "Nano drones, and micro drones operated for non-commercial purposes.",
        R21,
        "page 32",
        "operating a micro unmanned aircraft system for non-commercial purposes",
    ),
    (
        "dr-11",
        "Within how many hours must a drone accident be reported, and to whom?",
        "No later than 48 hours, by the remote pilot to the Director General through the digital sky platform.",
        R21,
        "page 31",
        "No later than forty-eight hours after an accident",
    ),
    (
        "dr-12",
        "What is the maximum penalty for contravening the Drone Rules 2021?",
        "A penalty not exceeding rupees one lakh (under section 10A of the Aircraft Act).",
        R21,
        "page 35",
        "levy a penalty not exceeding rupees one lakh",
    ),
    (
        "dr-13",
        "What is the fee for issuance or renewal of a remote pilot licence?",
        "Rs 100.",
        R21,
        "page 34",
        "Issuance or renewal of remote pilot licence 100",
    ),
    (
        "dr-14",
        "What fee is charged for authorising a remote pilot training organisation?",
        "Rs 1,000 for authorisation or renewal of authorisation of an RPTO.",
        R21,
        "page 34",
        "Authorisation or renewal of authorisation of remote pilot training organisation 1,000",
    ),
    (
        "dr-15",
        "Which drones do not need a type certificate to be operated?",
        "A model remotely piloted aircraft system and a nano unmanned aircraft system.",
        R21,
        "page 29",
        "No type certificate shall be required for operating the following",
    ),
    (
        "dr-16",
        "How long can a temporary red zone be declared for, and who can declare it?",
        "Up to 96 hours at a time, declared by the State Government/UT Administration or a law enforcement agency through an officer not below the rank of Superintendent of Police.",
        R21,
        "page 31",
        "exceeding ninety six hours at a time",
    ),
    (
        "dr-17",
        "Within how many days must the new serial number be updated on Digital Sky after replacing a drone's flight control module?",
        "Within seven days of the replacement or before operating the drone, whichever is earlier.",
        R21,
        "page 29",
        "within a period of seven days from the date of such replacement",
    ),
    (
        "dr-18",
        "What is the maximum all-up weight of a model remotely piloted aircraft system?",
        "Not exceeding twenty-five kilograms, used for educational, research, design, testing or recreational purposes within visual line of sight.",
        R21,
        "page 26",
        "with all -up\nweight not exceeding twenty -five kilograms",
    ),
    (
        "dr-19",
        "Which form is used to apply for a type certificate?",
        "Form D-1 on the digital sky platform.",
        R21,
        "page 28",
        "shall make an application in Form D -1 on the digital sky platform",
    ),
    (
        "dr-20",
        "Do the Drone Rules 2021 apply to drones used by the Indian armed forces?",
        "No. They do not apply to UAS belonging to or used by the naval, military or air forces of the Union of India.",
        R21,
        "page 25",
        "These rules shall not apply to an unmanned aircraft system belonging to, or used by , the naval",
    ),
    (
        "dr-21",
        "Which mandatory safety features may the Central Government notify for drones?",
        "Features such as 'No Permission - No Takeoff' hardware and firmware, a real-time tracking beacon, and geo-fencing capability.",
        R21,
        "page 29",
        "'No Permission - No Takeoff' hardware and firmware",
    ),
    (
        "dr-22",
        "Do research and development entities need a remote pilot licence to test drones?",
        "No. Listed R&D entities, recognised educational institutions, DPIIT startups, authorised testing entities and manufacturers with GSTIN need no type certificate, UIN, prior permission or remote pilot licence, provided operations are within a green zone on their premises or an open area under their control.",
        R21,
        "page 33",
        "shall not require a type certificate, unique identification number, prior permission and remote pilot\nlicence",
    ),
    (
        "dr-23",
        "Is third party insurance mandatory for nano drones?",
        "No. A nano unmanned aircraft system may operate without third party insurance.",
        R21,
        "page 33",
        "a nano unmanned aircraft system may operate without third party insurance",
    ),
    # ---------------- Amendments ----------------
    (
        "am-01",
        "What did the Drone (Amendment) Rules 2022 replace the remote pilot licence with?",
        "A Remote Pilot Certificate issued by an authorised remote pilot training organisation under rule 34; 'licence' was replaced by 'certificate'.",
        A22,
        "page 2",
        'Remote Pilot Certificate" means the certificate issued by an authorised remote pilot training',
    ),
    (
        "am-02",
        "When were the Drone (Amendment) Rules 2022 notified?",
        "On 11 February 2022 (G.S.R. 108(E)).",
        A22,
        "page 2",
        "New Delhi, the 11th February, 2022",
    ),
    (
        "am-03",
        "What ID documents does the Drone (Amendment) Rules 2023 allow in Form D-4 instead of only a passport?",
        "Indian passport number or a government-issued proof of identity and proof of address, such as a Voter ID card, Ration Card or Driving Licence.",
        A23,
        "page 2",
        "Indian Passport Number or government issued proof of identity and government issued proof of address",
    ),
    # ---------------- PIB backgrounder ----------------
    (
        "pib-01",
        "By how much did the Drone Rules 2021 reduce the number of forms and fee types?",
        "Forms were reduced from 25 to 5 and types of fee from 72 to 4.",
        PIB,
        "page 3",
        "Number of forms reduced from 25 to 5",
    ),
    (
        "pib-02",
        "How far from the airport perimeter does the yellow zone extend after the 2021 rules?",
        "The yellow zone was reduced from 45 km to 12 km from the airport perimeter.",
        PIB,
        "page 3",
        "has been reduced from 45\nkm to 12 km from the airport perimeter",
    ),
    # ---------------- ArduPilot Copter ----------------
    (
        "ap-01",
        "How do you arm an ArduPilot copter using the sticks?",
        "Hold the throttle down and rudder right for 5 seconds (not longer than 15 s or AutoTrim starts).",
        AP_ARM,
        "section: Arming the motors",
        "Arm the motors by holding the throttle down, and rudder right for 5 seconds",
    ),
    (
        "ap-02",
        "In ArduPilot Copter, what happens if the throttle is left at minimum for 15 seconds after arming?",
        "The motors will automatically disarm.",
        AP_ARM,
        "section: Arming the motors",
        "If you leave the throttle at minimum for 15 seconds",
    ),
    (
        "ap-03",
        "What is the default BATT_LOW_VOLT battery failsafe voltage in ArduPilot Copter and how long must voltage stay below it?",
        "10.5 volts; the voltage must stay below it for more than 10 seconds.",
        AP_BATT,
        "section: Battery Failsafe > When the failsafe will trigger",
        "The default voltage is 10.5 volts",
    ),
    (
        "ap-04",
        "What does BATT_FS_LOW_ACT = 2 do in ArduPilot?",
        "RTL: the vehicle switches to Return-to-Launch if possible, otherwise it lands (the recommended option).",
        AP_BATT,
        "section: Battery Failsafe > What will happen",
        "2 : RTL - the vehicle will switch to Return-to-Launch mode if possible",
    ),
    (
        "ap-05",
        "What is the default RC_FS_TIMEOUT for ArduPilot radio failsafe?",
        "1 second.",
        AP_RADIO,
        "section: Radio Failsafe > When the failsafe will trigger",
        "RC_FS_TIMEOUT seconds (default = 1 sec)",
    ),
    (
        "ap-06",
        "What does FS_THR_ENABLE value 1 do in ArduPilot Copter?",
        "Enabled Always RTL: switches to RTL mode, or to Land if the GPS position is not usable.",
        AP_RADIO,
        "section: Radio Failsafe > Parameter Configuration",
        "Enabled Always RTL (Value 1) will switch the copter to RTL Mode",
    ),
    (
        "ap-07",
        "What is the default RTL_ALT_M return altitude in ArduPilot Copter?",
        "15 metres.",
        AP_RTL,
        "section: RTL Mode > Overview",
        "The default value for RTL_ALT_M is 15m",
    ),
    (
        "ap-08",
        "What does RTL_CONE_SLOPE control and what is its default?",
        "It sets the slope of an inverted cone around the return point that limits how much the copter climbs when RTL starts close to home; default slope is 3.",
        AP_RTL,
        "section: RTL Mode > Overview",
        "with the default slope of\n3",
    ),
    (
        "ap-09",
        "What is the default final descent speed LAND_SPD_MS in ArduPilot Land mode?",
        "50 cm/s.",
        AP_LAND,
        "section: Land Mode",
        "LAND_SPD_MS parameter which defaults to 50cm/s",
    ),
    (
        "ap-10",
        "How many built-in flight modes does ArduPilot Copter have?",
        "25 built-in flight modes, 10 of which are regularly used.",
        AP_MODES,
        "section: Flight Modes > Overview",
        "Copter has 25 flight built-in flight modes, 10 of which are regularly",
    ),
    (
        "ap-11",
        "Which ArduPilot copter flight modes do not require position information?",
        "Acro, AirMode, Alt Hold, Stabilize, Sport, SysID, Turtle and Land.",
        AP_MODES,
        "section: Flight Modes > GPS Dependency",
        "Do not require position information:",
    ),
    # ---------------- PX4 ----------------
    (
        "px-01",
        "What are the default PX4 arming and disarming stick gestures?",
        "Arm: throttle minimum, yaw maximum; disarm: throttle minimum, yaw minimum; held for 1 second.",
        PX_ARM,
        "section: Arm, Disarm, Prearm Configuration > Arming/Disarming Gestures",
        "Throttle minimum, yaw maximum",
    ),
    (
        "px-02",
        "What is the default PX4 auto-disarm timeout after landing (COM_DISARM_LAND)?",
        "2 seconds (-1 disables it).",
        PX_ARM,
        "section: Arm, Disarm, Prearm Configuration > Auto-Disarming",
        "Time-out for auto disarm after landing. Default: 2s",
    ),
    (
        "px-03",
        "How long does PX4 wait for takeoff before auto-disarming (COM_DISARM_PRFLT)?",
        "10 seconds by default (-1 disables it).",
        PX_ARM,
        "section: Arm, Disarm, Prearm Configuration > Auto-Disarming",
        "Time-out for auto disarm if too slow to takeoff. Default: 10s",
    ),
    (
        "px-04",
        "At what rate does PX4 run arming checks internally?",
        "At 10 Hz.",
        PX_ARM,
        "section: Arm, Disarm, Prearm Configuration > Pre-Arm Checks",
        "internally PX4 runs arming checks at 10Hz",
    ),
    (
        "px-05",
        "In PX4, if several failsafes trigger at once, which action is taken?",
        "The more severe action is taken (e.g. Land over Return).",
        PX_SAFE,
        "section: Safety (Failsafe) Configuration > Failsafe Actions",
        "If multiple failsafes are triggered, the more severe action is taken",
    ),
    (
        "px-06",
        "Which PX4 parameter sets the geofence breach action, and what options does it have?",
        "GF_ACTION: None, Warning, Hold mode, Return mode, Terminate, Land.",
        PX_SAFE,
        "section: Safety (Failsafe) Configuration > Geofence Failsafe",
        "None, Warning, Hold mode, Return mode, Terminate, Land.",
    ),
    (
        "px-07",
        "Which COM_LOW_BAT_ACT option does PX4 recommend for the battery failsafe?",
        "Option 3: distance-aware return, critical level return, emergency level land.",
        PX_SAFE,
        "section: Safety (Failsafe) Configuration > Battery Failsafes > Battery level failsafe",
        "`3: Distance-aware return, Critical level return, Emergency level land` (recommended)",
    ),
    (
        "px-08",
        "What does the PX4 maximum flight time failsafe COM_FLT_TIME_MAX do?",
        "After the set flight time it engages Return mode (warning at 90%); -1 disables it (default).",
        PX_SAFE,
        "section: Safety (Failsafe) Configuration > Battery Failsafes > Maximum Flight Time Failsafe",
        'it will also "warn" at 90% of this time',
    ),
    (
        "px-09",
        "In PX4 Position mode, above what throttle stick position does the vehicle take off when landed?",
        "Above 62.5% of the full stick range from the bottom.",
        PX_POS,
        "section: Position Mode (Multicopter) > Technical Summary",
        "raised above 62.5% percent",
    ),
    (
        "px-10",
        "What are the default maximum ascent and descent velocities in PX4 Position mode?",
        "MPC_Z_VEL_MAX_UP 3 m/s ascent and MPC_Z_VEL_MAX_DN 1 m/s descent.",
        PX_POS,
        "section: Position Mode (Multicopter) > Technical Summary > Parameters",
        "Maximum vertical ascent velocity. Default: 3 m/s",
    ),
    (
        "px-11",
        "Which parameter sets the descent rate in PX4 multicopter Land mode?",
        "MPC_LAND_SPEED.",
        PX_LAND,
        "section: Land Mode (Multicopter) > Technical Summary",
        "The vehicle descends at the rate specified in MPC_LAND_SPEED",
    ),
    (
        "px-12",
        "What position estimate does PX4 Return mode require?",
        "A global 3D position estimate (from GPS or inferred from a local position), and the home position must be set.",
        PX_RTN,
        "section: Return Mode (Generic Vehicle)",
        "Mode requires a global 3d position estimate",
    ),
    (
        "px-13",
        "What is the default PX4 Return mode behaviour for a multicopter?",
        "Home/rally point return.",
        PX_RTN,
        "section: Return Mode (Generic Vehicle)",
        "Multicopter - Home/rally point return",
    ),
]

# Some facts appear in more than one document (the PIB backgrounder restates the Drone Rules).
# A citation to any of these manually-verified alternate passages counts as correct.
ALT_EVIDENCE = {
    "dr-01": [(PIB, "page 1", "Nano unmanned aircraft system: weighing less than or equal to 250 grams")],
    "dr-02": [
        (
            PIB,
            "page 1",
            "Micro unmanned aircraft system: weighing more than 250 grams, but less\nthan or equal to Two kilograms",
        )
    ],
    "dr-03": [
        (
            PIB,
            "page 1",
            "Medium unmanned aircraft system : weighing more than 25 kilograms, but\nless than or equal to 150 kilograms",
        )
    ],
    "dr-04": [(PIB, "page 2", "Large unmanned aircraft system: weighing more than 150 kilograms")],
    "dr-06": [(PIB, "page 3", "No permission required for operating drones in green zones")],
    "dr-09": [(PIB, "page 3", "INR 100 for all categories of drones; and is valid for 10")],
    "dr-10": [(PIB, "page 3", "No remote pilot licence required for micro drones (for non -commercial use)")],
    "dr-13": [(PIB, "page 3", "INR 100 for all categories of drones")],
    "dr-15": [(PIB, "page 4", "Nano and model drones (made for research or recreation purposes) are exempt")],
    "dr-21": [(PIB, "page 4", "'No Permission - No Takeoff' (NPNT), real-")],
    "dr-22": [(PIB, "page 3", "No requirement of Type Certificate, unique identification number and remote")],
}

# Questions whose answer is NOT in the corpus -> the system must refuse.
UNANSWERABLE = [
    ("na-01", "What is the drone registration fee in the United States under FAA Part 107?"),
    ("na-02", "What is the maximum takeoff weight of the DJI Mavic 3 Pro?"),
    ("na-03", "How do I tune the PID gains of a Betaflight racing quad?"),
    ("na-04", "What is the price of a DGCA remote pilot training course in Hyderabad?"),
    ("na-05", "Which EU drone class is C2 under EASA regulations?"),
    ("na-06", "What is the default MPC_XY_CRUISE speed in PX4 missions?"),
    ("na-07", "How many drone pilots were certified in India in 2025?"),
    ("na-08", "What LiPo cell count does ArduPilot recommend for a 450 mm quadcopter?"),
]


def build() -> list[dict]:
    rows = []
    for qid, q, ref, src, loc, ev in ANSWERABLE:
        alts = [{"source": a, "locator": b, "evidence": c} for a, b, c in ALT_EVIDENCE.get(qid, [])]
        rows.append(
            {
                "id": qid,
                "question": q,
                "reference": ref,
                "answerable": True,
                "source": src,
                "locator": loc,
                "evidence": ev,
                "alt_evidence": alts,
            }
        )
    for qid, q in UNANSWERABLE:
        rows.append(
            {
                "id": qid,
                "question": q,
                "reference": "INSUFFICIENT_EVIDENCE",
                "answerable": False,
                "source": None,
                "locator": None,
                "evidence": None,
            }
        )
    return rows


if __name__ == "__main__":
    out = Path(__file__).with_name("golden.jsonl")
    rows = build()
    out.write_text("".join(json.dumps(r) + "\n" for r in rows))
    print(f"wrote {len(rows)} items ({sum(r['answerable'] for r in rows)} answerable) -> {out}")
