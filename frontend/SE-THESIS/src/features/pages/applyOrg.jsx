import { useState, useEffect } from "react";
import Popover from "@mui/material/Popover";
import { toast } from "sonner";
import { addOrganization, getOrganization } from "../../shared/services/organization";
import { updateOrganization } from "../../shared/services/authService.js";
import { useAuth } from "../../context/authContext.jsx";

export default function ApplyOrg() {
  const { user } = useAuth();
  const [orgName, setOrgName] = useState("");
  const [organizations, setOrganizations] = useState([]);
  const [anchorEl, setAnchorEl] = useState(null);
  const [selectedOrg, setSelectedOrg] = useState(null);
  const [selectedOrgName, setSelectedOrgName] = useState(null);

  useEffect(() => {
    const fetchOrganizations = async () => {
      try {
        const orgs = await getOrganization();
        setOrganizations(orgs);
      } catch (error) {
        console.error("Error fetching organizations:", error);
      }
    };
    fetchOrganizations();
  }, []);

  const handleApply = async () => {
    if (user?.is_authorized) {
      toast.error("You're already in an organization.");
      return;
    }
    if (!selectedOrgName) return;
    try {
      await updateOrganization(selectedOrgName);
      toast.success("Organization applied successfully!");
    } catch (error) {
      console.error("Error applying organization:", error);
      toast.error("Failed to apply organization.");
    }
  };

  const handleClick = (event) => {
    setAnchorEl(event.currentTarget);
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    try {
      await addOrganization({ org_name: orgName });
      setOrgName("");
      handleClose();
      
      const orgs = await getOrganization();
      setOrganizations(orgs);
      toast.success("Organization added successfully!");
    } catch (error) {
      console.error("Error adding organization:", error);
      toast.error("Failed to add organization.");
    }
  };

  const handleClose = () => {
    setAnchorEl(null);
  };

  const open = Boolean(anchorEl);
  const id = open ? "simple-popover" : undefined;

  return (    
    <div className="w-full h-full flex flex-col gap-6 p-8">
      <div className="flex flex-col gap-4 flex-1">
        <input
          type="text"
          placeholder="Search organizations..."
          className="w-full h-9 px-4 py-1 text-base border border-[#d4d3d1] rounded-full shadow-inner focus:outline-none focus:border-[#858585] bg-[#E4E3E1]"
        />
        <div className="flex-1 bg-[#E4E3E1] border border-[#d4d3d1] rounded-xl shadow-inner overflow-y-auto">
          {organizations.map((org) => (
            <div
              key={org._id}
              onClick={() => {
                setSelectedOrg(org._id);
                setSelectedOrgName(org.organization_name);
              }}
              className={`py-2 px-3 text-lg cursor-pointer rounded transition-colors duration-150
                ${selectedOrg === org._id ? "bg-[#7E808C] text-white" : "hover:bg-[#A7A7A3]"}
              `}
            >
              {org.organization_name}
            </div>
          ))}
        </div>
        <div className="flex flex-row gap-4 justify-end">
          <button 
            className="w-62 h-10 bg-[#A1A2A6] text-white px-6 rounded-lg text-base hover:bg-[#7E808C] transition-colors duration-300"
            onClick={handleClick}
          >
            Add Organization
          </button>
          <button
            onClick={handleApply}
            className="w-62 h-10 bg-[#A1A2A6] text-white px-6 rounded-lg text-base hover:bg-[#7E808C] transition-colors duration-300"
          >
            Apply
          </button>
        </div>
      </div>
      <Popover
        id={id}
        open={open}
        anchorEl={anchorEl}
        onClose={handleClose}
        slotProps={{
          paper: {
            sx: {
              backgroundColor: "#DFDEDA",
              borderRadius: "15px",
              marginY: -2,
            },
          },
        }}
        anchorOrigin={{
          vertical: "top",
          horizontal: "center",
        }}
        transformOrigin={{
          vertical: "bottom",
          horizontal: "center",
        }}
      >
        <div className="w-84 rounded-lg flex flex-col gap-2 p-5 bg-[#DFDEDA]">
          <h2 className="text-lg text-[#4F4F4F]">Add Organization</h2>
          <form onSubmit={handleSubmit} className="flex flex-col gap-3">
            <input
            type="text"
            value={orgName}
            onChange={(e) => setOrgName(e.target.value)}
            placeholder="Enter Organization Name"
            className="w-full min-w-0 bg-[#E4E3E1] rounded-full px-4 py-2 text-base focus:outline-none"
          />
            <button
              type="submit"
              className="bg-[#A1A2A6] text-white py-2 rounded-lg text-base hover:bg-[#7E808C] transition-colors duration-300"
            >
              Add
            </button>
          </form>
        </div>
      </Popover>
    </div>
  );
}