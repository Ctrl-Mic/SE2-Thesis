import { X } from "lucide-react";

export default function EditSchedule({ open, onClose, roomId }) {
  if (!open) return null;

  return (
    <div className="fixed inset-0 z-50">
      {/* Dark overlay backdrop */}
      <div
        className="absolute inset-0 bg-black/50 transition-opacity duration-300"
        onClick={onClose}
      />

      {/* Modal container */}
      <div className="absolute inset-0 flex items-center justify-center pointer-events-none">
        <div className="pointer-events-auto bg-[#DFDEDA] rounded-lg shadow-2xl overflow-hidden w-[85vw] h-[85vh] max-w-6xl max-h-[90vh] flex flex-col">
          {/* Header */}
          <div className="bg-[#C4C3C0] shadow-md px-8 py-6 flex items-center justify-between border-b border-black/10">
            <h2 className="text-2xl font-bold text-[#4F4F4F]">Edit Schedule</h2>
            <button
              onClick={onClose}
              className="cursor-pointer hover:scale-110 transition-transform duration-150 p-2 rounded-full hover:bg-black/10"
            >
              <X size={24} color="#4F4F4F" />
            </button>
          </div>

          {/* Content area - blank for now */}
          <div className="flex-1 bg-[#DFDEDA] overflow-y-auto p-8">
            {/* Content will be added here */}
          </div>
        </div>
      </div>
    </div>
  );
}
