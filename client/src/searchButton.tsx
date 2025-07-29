import React, { useState } from "react";
import { Search } from "lucide-react";
import SearchModal from "./searchModal";

export default function SearchButton() {
  const [showModal, setShowModal] = useState(false);

  return (
    <>
      <button
        onClick={() => setShowModal(true)}
        className="ml-2 p-2 rounded hover:bg-gray-100 text-gray-600"
        title="Search PDFs"
      >
        <Search className="w-5 h-5" />
      </button>

      {showModal && <SearchModal onClose={() => setShowModal(false)} />}
    </>
  );
}
