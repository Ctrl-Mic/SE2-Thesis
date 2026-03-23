import { Edit2 } from "lucide-react";
import Popover from "@mui/material/Popover";

export default function KebabPullout({
  open,
  anchorEl,
  onClose,
  onEditClassroom,
  onEditSchedule,
}) {
  const id = open ? "kebab-popover" : undefined;

  const handleEditClassroom = () => {
    onEditClassroom();
    onClose();
  };

  const handleEditSchedule = () => {
    onEditSchedule();
    onClose();
  };

  return (
    <Popover
      id={id}
      open={open}
      anchorEl={anchorEl}
      onClose={onClose}
      slotProps={{
        paper: {
          sx: {
            backgroundColor: "#DFDEDA",
            borderRadius: "8px",
            boxShadow: "0 4px 12px rgba(0, 0, 0, 0.15)",
          },
        },
      }}
      anchorOrigin={{
        vertical: "center",
        horizontal: "right",
      }}
      transformOrigin={{
        vertical: "center",
        horizontal: "left",
      }}
    >
      <div className="bg-[#DFDEDA] rounded-lg overflow-hidden min-w-[200px]">
        <button
          onClick={handleEditClassroom}
          className="w-full px-6 py-3 flex items-center gap-3 text-[#4F4F4F] hover:bg-[#C4C3C0] transition-colors duration-150 text-subtitle font-medium"
        >
          <Edit2 size={18} />
          Edit Classroom
        </button>
        <button
          onClick={handleEditSchedule}
          className="w-full px-6 py-3 flex items-center gap-3 text-[#4F4F4F] hover:bg-[#C4C3C0] transition-colors duration-150 text-subtitle font-medium border-t border-[#C4C3C0]"
        >
          <Edit2 size={18} />
          Edit Schedule
        </button>
      </div>
    </Popover>
  );
}
