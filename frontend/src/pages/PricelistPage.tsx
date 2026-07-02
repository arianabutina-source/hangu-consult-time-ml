const priceList = [
  {
    category: "Dermatology",
    items: [
      { service: "Specialist Consultation", price: "150" },
      { service: "Follow-up Consultation", price: "100" },
      { service: "Mole Check (Dermoscopy)", price: "120" },
      { service: "Full Body Mole Mapping", price: "250" },
      { service: "Skin Cancer Screening", price: "180" },
      { service: "Acne / Rosacea / Hair Consultation", price: "150" },
      { service: "Cryotherapy (per lesion)", price: "50" },
      { service: "Skin Biopsy", price: "300" },
      { service: "Minor Skin Surgery", price: "500+" },
      { service: "Histopathology", price: "100" },
    ],
  },
  {
    category: "Aesthetic Medicine",
    items: [
      { service: "Aesthetic Consultation*", price: "100" },
      { service: "Botulinum Toxin – 1 Area", price: "300" },
      { service: "Botulinum Toxin – 2 Areas", price: "450" },
      { service: "Botulinum Toxin – 3 Areas", price: "600" },
      { service: "Full Face Botox", price: "800" },
      { service: "Hyperhidrosis Treatment", price: "900" },
      { service: "Lip Augmentation", price: "500" },
      { service: "Cheek Enhancement", price: "700" },
      { service: "Chin Enhancement", price: "600" },
      { service: "Jawline Contouring", price: "900" },
      { service: "Tear Trough Correction", price: "700" },
      { service: "Full Facial Rejuvenation (Fillers)", price: "2,000+" },
      { service: "Skin Booster", price: "450" },
      { service: "Profhilo®", price: "500" },
      { service: "Polynucleotides", price: "500" },
      { service: "PRP Face", price: "500" },
      { service: "PRP Hair", price: "450" },
      { service: "Sculptra®", price: "900" },
      { service: "Radiesse®", price: "800" },
      { service: "Chemical Peel", price: "200" },
      { service: "Microneedling", price: "350" },
      { service: "Microneedling + PRP", price: "700" },
      { service: "Medical Facial", price: "250" },
      { service: "HydraFacial®", price: "300" },
      { service: "Pigmentation Laser", price: "350+" },
      { service: "Vascular Laser", price: "350+" },
      { service: "Fractional Laser Resurfacing", price: "800+" },
      { service: "Tattoo Removal", price: "250+" },
      { service: "Laser Hair Removal", price: "150–900" },
    ],
  },
  {
    category: "Plastic Surgery",
    items: [
      { service: "Plastic Surgery Consultation*", price: "150" },
      { service: "Mole / Skin Lesion Removal", price: "700+" },
      { service: "Lipoma / Cyst Removal", price: "900+" },
      { service: "Scar Revision", price: "1,200+" },
      { service: "Earlobe Repair", price: "800" },
      { service: "Upper Blepharoplasty", price: "3,500" },
      { service: "Lower Blepharoplasty", price: "4,000" },
      { service: "Upper & Lower Blepharoplasty", price: "6,500" },
      { service: "Otoplasty", price: "4,500" },
      { service: "Rhinoplasty", price: "9,000" },
      { service: "Facelift", price: "15,000" },
      { service: "Neck Lift", price: "10,000" },
      { service: "Mini Facelift", price: "9,000" },
      { service: "Breast Augmentation", price: "8,000" },
      { service: "Breast Lift", price: "9,000" },
      { service: "Breast Reduction", price: "10,000" },
      { service: "Liposuction", price: "5,000+" },
      { service: "Abdominoplasty", price: "10,000" },
    ],
  },
];

export function PricelistPage() {
  return (
    <div className="mx-auto max-w-3xl px-6 py-14">
      <span className="text-xs font-semibold tracking-wide text-terracotta uppercase">
        Pricelist
      </span>
      <h1 className="mt-2 font-serif text-3xl font-medium text-espresso sm:text-4xl">
        Service Pricing
      </h1>

      <div className="mt-8 space-y-10">
        {priceList.map((group) => (
          <section key={group.category}>
            <h2 className="mb-3 font-serif text-xl font-medium text-terracotta-dark uppercase">
              {group.category}
            </h2>
            <div className="overflow-hidden rounded-xl border border-espresso/5 bg-white shadow-sm">
              <table className="w-full text-left text-sm">
                <thead>
                  <tr className="border-b border-espresso/10 bg-terracotta-light">
                    <th className="px-5 py-3 font-serif font-medium text-espresso">Service</th>
                    <th className="px-5 py-3 text-right font-serif font-medium text-espresso">
                      Price (€)
                    </th>
                  </tr>
                </thead>
                <tbody>
                  {group.items.map((item, index) => (
                    <tr
                      key={item.service}
                      className={index % 2 === 0 ? "bg-white" : "bg-cream/40"}
                    >
                      <td className="px-5 py-2.5 text-espresso-light">{item.service}</td>
                      <td className="px-5 py-2.5 text-right font-medium text-espresso">
                        {item.price}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </section>
        ))}
      </div>
    </div>
  );
}
